//! Framed append-only WAL for [`Event`]s with CRC32 and O(1) checkpoints.
//!
//! Frame format: `[u8 marker][u32 len LE][u32 crc32 LE][payload: len bytes]`
//! - marker 0x01: postcard-serialized [`Event`]
//! - marker 0x02: postcard-serialized [`Checkpoint`] envelope

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::domain::{Event, State};
use crate::reduce;

const MARKER_EVENT: u8 = 0x01;
const MARKER_CHECKPOINT: u8 = 0x02;
const BATCH_FLUSH_EVERY: u64 = 64;

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u64,
    pub state: State,
}

pub struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
    events_since_flush: u64,
}

fn write_frame(w: &mut BufWriter<File>, marker: u8, payload: &[u8]) -> std::io::Result<()> {
    w.write_all(&[marker])?;
    w.write_all(&(payload.len() as u32).to_le_bytes())?;
    w.write_all(&crc32fast::hash(payload).to_le_bytes())?;
    w.write_all(payload)
}

impl Wal {
    /// Opens or creates the WAL file (append mode).
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
            events_since_flush: 0,
        })
    }

    /// Appends an event buffered; flushes at least every 64th event.
    pub fn append(&mut self, event: &Event) -> std::io::Result<()> {
        let payload = postcard::to_allocvec(event)
            .map_err(|_| std::io::Error::other("postcard: event serialization failed"))?;
        write_frame(&mut self.writer, MARKER_EVENT, &payload)?;
        self.events_since_flush += 1;
        if self.events_since_flush >= BATCH_FLUSH_EVERY {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush of the buffer (batch commit durability unit).
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.events_since_flush = 0;
        Ok(())
    }

    /// Writes a checkpoint record (marker 0x02) and flushes immediately.
    pub fn checkpoint(&mut self, state: &State) -> std::io::Result<()> {
        let envelope = Checkpoint {
            version: state.version,
            state: state.clone(),
        };
        let payload = postcard::to_allocvec(&envelope)
            .map_err(|_| std::io::Error::other("postcard: checkpoint serialization failed"))?;
        write_frame(&mut self.writer, MARKER_CHECKPOINT, &payload)?;
        self.flush()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Reads the next full frame. Returns `Ok(None)` on clean EOF.
/// Returns `Ok(None)` (after `consumed` is left at frame start) on a broken
/// tail; the caller discards from the frame start onward.
fn read_frame<R: Read>(r: &mut R) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut header = [0u8; 9];
    match r.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let marker = header[0];
    let len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let crc = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
    if len > crate::wal::MAX_FRAME_LEN {
        return Ok(None); // implausible length: treat as corrupt tail
    }
    let mut payload = vec![0u8; len];
    if r.read_exact(&mut payload).is_err() {
        return Ok(None); // truncated frame: corrupt tail
    }
    if crc32fast::hash(&payload) != crc {
        return Ok(None); // CRC mismatch: corrupt tail
    }
    if marker != MARKER_EVENT && marker != MARKER_CHECKPOINT {
        return Ok(None); // unknown marker: corrupt tail
    }
    Ok(Some((marker, payload)))
}

/// Recovery: reads the last valid checkpoint, then all valid event records.
/// A corrupt/incomplete tail (CRC error, torn frame) is silently discarded;
/// the valid prefix survives.
pub fn recover(path: &Path) -> std::io::Result<(Option<State>, Vec<Event>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let mut cursor = &bytes[..];

    let mut state: Option<State> = None;
    let mut events = Vec::new();
    while let Some((marker, payload)) = read_frame(&mut cursor)? {
        match marker {
            MARKER_CHECKPOINT => match postcard::from_bytes::<Checkpoint>(&payload) {
                Ok(cp) => {
                    state = Some(cp.state);
                    events.clear();
                }
                Err(_) => break,
            },
            _ => match postcard::from_bytes::<Event>(&payload) {
                Ok(event) => events.push(event),
                Err(_) => break,
            },
        }
    }
    Ok((state, events))
}

/// Plays events onto a state via the pure reducer; returns the final state.
pub fn replay(state: Option<State>, events: &[Event]) -> State {
    let mut state = state.unwrap_or_else(|| State::new(Default::default()));
    for event in events {
        let (next, _effects) = reduce(&state, event);
        state = next;
    }
    state
}

const MAX_FRAME_LEN: usize = 1 << 30;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events(n: usize) -> Vec<Event> {
        (0..n)
            .map(|i| match i % 3 {
                0 => Event::TaskSubmitted {
                    goal: format!("goal-{i}"),
                },
                1 => Event::ArtifactInstalled {
                    playbook_name: format!("pb-{i}"),
                    content: format!("content-{i}-payload"),
                },
                _ => Event::RollbackRequested {
                    target_version: i as u64,
                },
            })
            .collect()
    }

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join("wal-test-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wal.log");
        let _ = std::fs::remove_file(&path);

        let mut wal = Wal::open(&path).unwrap();
        let events = sample_events(100);
        for event in &events {
            wal.append(event).unwrap();
        }
        wal.flush().unwrap();

        let (state, recovered) = recover(&path).unwrap();
        assert!(state.is_none());
        assert_eq!(events.len(), recovered.len());
        for (original, got) in events.iter().zip(&recovered) {
            assert_eq!(
                postcard::to_stdvec(original).unwrap(),
                postcard::to_stdvec(got).unwrap()
            );
        }
    }

    #[test]
    fn checkpoint_recovery() {
        let dir = std::env::temp_dir().join("wal-test-checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wal.log");
        let _ = std::fs::remove_file(&path);

        let first = sample_events(30);
        let second: Vec<Event> = sample_events(30)
            .into_iter()
            .map(|e| match e {
                Event::TaskSubmitted { goal } => Event::TaskSubmitted {
                    goal: format!("{goal}-b"),
                },
                other => other,
            })
            .collect();

        let mut wal = Wal::open(&path).unwrap();
        for event in &first {
            wal.append(event).unwrap();
        }
        let cp_state = replay(None, &first);
        wal.checkpoint(&cp_state).unwrap();
        for event in second.iter() {
            wal.append(event).unwrap();
        }
        wal.flush().unwrap();

        let (state, events) = recover(&path).unwrap();
        assert_eq!(state.unwrap().hash(), cp_state.hash());
        assert_eq!(events.len(), 30);
        for (original, got) in second.iter().zip(&events) {
            assert_eq!(
                postcard::to_stdvec(&original).unwrap(),
                postcard::to_stdvec(got).unwrap()
            );
        }
    }

    #[test]
    fn tail_corruption() {
        let dir = std::env::temp_dir().join("wal-test-corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wal.log");
        let _ = std::fs::remove_file(&path);

        let events = sample_events(10);
        let mut wal = Wal::open(&path).unwrap();
        for event in &events {
            wal.append(event).unwrap();
        }
        wal.flush().unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 3); // tear the last frame
        std::fs::write(&path, &bytes).unwrap();

        let (state, recovered) = recover(&path).unwrap();
        assert!(state.is_none());
        assert_eq!(recovered.len(), events.len() - 1);

        // State must equal reference replay without the last event.
        let expected = replay(None, &events[..events.len() - 1]);
        let got = replay(state, &recovered);
        assert_eq!(got.hash(), expected.hash());
    }

    #[test]
    fn replay_determinism() {
        let events = sample_events(50);
        let a = replay(None, &events);
        let b = replay(None, &events);
        assert_eq!(a.hash(), b.hash());
    }
}

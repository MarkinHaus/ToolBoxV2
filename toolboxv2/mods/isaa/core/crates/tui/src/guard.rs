//! RAII-Guard: stellt Terminal-Zustand via `Drop` IMMER wieder her
//! (`LeaveAlternateScreen` + `cursor::Show`), auch bei Panic/Unwind.

use std::io::{self, Write};

type RestoreFn = Box<dyn FnOnce() + Send>;

pub struct TerminalGuard {
    restore: Option<RestoreFn>,
}

impl TerminalGuard {
    /// Echtes Terminal: Raw Mode + Alternate Screen + versteckter Cursor.
    #[allow(clippy::new_without_default)] // Default würde Terminal-State global ändern
    pub fn new() -> io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        let mut out = io::stdout();
        if let Err(e) = crossterm::execute!(
            out,
            crossterm::terminal::EnterAlternateScreen,
            crossterm::cursor::Hide
        ) {
            let _ = crossterm::terminal::disable_raw_mode();
            return Err(e);
        }
        Ok(Self::with_restore(
            || Ok(()),
            || {
                let _ = crossterm::terminal::disable_raw_mode();
                let mut o = io::stdout();
                let _ = crossterm::execute!(
                    o,
                    crossterm::terminal::LeaveAlternateScreen,
                    crossterm::cursor::Show
                );
                let _ = o.flush();
            },
        ))
    }

    /// Injektierbarer Einstieg für Tests/Fake-Terminal: `enter` läuft beim Bau,
    /// `restore` läuft garantiert im `Drop` (auch bei Panic im Owner-Frame).
    pub fn with_restore<E, R>(enter: E, restore: R) -> Self
    where
        E: FnOnce() -> io::Result<()>,
        R: FnOnce() + Send + 'static,
    {
        if let Err(e) = enter() {
            tracing::warn!("terminal enter failed: {e}");
        }
        Self {
            restore: Some(Box::new(restore)),
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if let Some(restore) = self.restore.take() {
            restore();
        }
    }
}

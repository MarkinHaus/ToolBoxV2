import logging
import os

import numpy as np
# import pyaudio

from toolboxv2 import get_app, get_logger

logger = get_logger()


import asyncio
import contextlib
import queue

# Import needed for random debugging logs
import random
import sys
import tempfile
import threading
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor

import websockets
from groq import Groq

# For language detection; install via pip install langdetect
from langdetect import detect
from pydantic import BaseModel, Field

pyaudio = lambda :None

# Configuration
class Config:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 2048
    FORMAT = 0#pyaudio.paInt16
    CHANNELS = 1
    VAD_AGGRESSIVENESS = 3
    SILENCE_THRESHOLD = 2.2  # seconds
    THINKING_TIMEOUT = 2.5  # seconds for intelligent delay to detect user thinking
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Audio processing
    MAX_QUEUE_SIZE = 250
    MAX_BUFFER_SIZE = SAMPLE_RATE * 80  # 1:20 minute:seconds max buffer

    # TTS settings
    ENERGY_THRESHOLD = 75

    # Debug flag
    DEBUG = os.getenv("DEBUG", "0") == "1"

    # Add sliding window configuration
    SLIDING_WINDOW_SIZE = 12  # 12 seconds per window
    SLIDING_WINDOW_OVERLAP = 1.5  # 1.5 seconds overlap between windows
    SLIDING_WINDOW_ENABLED = True

    # Derived window sizes in bytes (16-bit audio)
    @classmethod
    def get_window_size_bytes(cls):
        return int(cls.SAMPLE_RATE * cls.SLIDING_WINDOW_SIZE * 2)

    @classmethod
    def get_window_overlap_bytes(cls):
        return int(cls.SAMPLE_RATE * cls.SLIDING_WINDOW_OVERLAP * 2)

    @classmethod
    def set_debug(cls, debug_mode):
        cls.DEBUG = debug_mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        else:
            logger.setLevel(logging.INFO)


# Audio Input Module
class AudioInputModule:
    def __init__(self, config, input_queue):
        self.config = config
        self.input_queue = input_queue
        self.running = False
        self.audio_interface = None
        self.stream = None
        self.websocket = None
        self.input_mode = None
        # Flag to mute input during TTS playback
        self.mute_input = False
        self.logger = logging.getLogger("SpeechSystem.AudioInput")

    async def start_microphone(self):
        """Initialize and start microphone input"""
        try:
            self.input_mode = "microphone"
            self.audio_interface = pyaudio.PyAudio()
            self.stream = self.audio_interface.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK_SIZE
            )
            self.running = True

            self.logger.info("Microphone input started")
            if self.config.DEBUG:
                self.logger.debug(f"Microphone config: rate={self.config.SAMPLE_RATE}, format={self.config.FORMAT}")

            # Start reading audio in a separate thread to not block the event loop
            threading.Thread(target=self._read_microphone, daemon=True).start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start microphone: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())
            return False

    def _read_microphone(self):
        """Read audio data from microphone in a separate thread"""
        error_count = 0
        while self.running:
            try:
                if self.stream is None:
                    self.logger.error("Microphone stream is None")
                    time.sleep(0.5)
                    continue

                data = self.stream.read(self.config.CHUNK_SIZE, exception_on_overflow=False)

                # Skip processing if muted (prevents TTS feedback)
                if not self.mute_input:
                    # Only add to queue if there's room to prevent memory issues
                    if self.input_queue.qsize() < self.config.MAX_QUEUE_SIZE:
                        self.input_queue.put(data)
                        if self.config.DEBUG and random.random() < 0.01:  # Log occasionally in debug mode
                            self.logger.debug(
                                f"Audio chunk queued: {len(data)} bytes, queue size: {self.input_queue.qsize()}")
                    else:
                        # Skip a frame if queue is full
                        if self.config.DEBUG:
                            self.logger.debug(f"Queue full ({self.input_queue.qsize()}), skipping audio frame")
                        time.sleep(0.01)
                elif not self.mute_input and self.config.DEBUG and random.random() < 0.01:
                    self.logger.debug("Audio input (TTS active)")

                # Reset error count on success
                error_count = 0
            except Exception as e:
                error_count += 1
                self.logger.error(f"Microphone read error: {e}")
                if self.config.DEBUG:
                    self.logger.debug(traceback.format_exc())

                # If we get too many consecutive errors, try to restart the stream
                if error_count > 10:
                    self.logger.warning("Too many microphone errors, attempting to restart stream")
                    try:
                        if self.stream:
                            self.stream.stop_stream()
                            self.stream.close()
                        self.stream = self.audio_interface.open(
                            format=self.config.FORMAT,
                            channels=self.config.CHANNELS,
                            rate=self.config.SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=self.config.CHUNK_SIZE
                        )
                        error_count = 0
                    except Exception as restart_error:
                        self.logger.error(f"Failed to restart microphone: {restart_error}")

                time.sleep(0.1)

    async def start_websocket(self, websocket_url):
        """Initialize and start websocket input"""
        try:
            self.input_mode = "websocket"
            self.running = True
            self.logger.info(f"Starting websocket connection to {websocket_url}")
            asyncio.create_task(self._read_websocket(websocket_url))
            return True
        except Exception as e:
            self.logger.error(f"Failed to start websocket: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())
            return False

    async def _read_websocket(self, websocket_url):
        """Read audio data from websocket"""
        retry_count = 0
        max_retries = 5

        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(websocket_url) as websocket:
                    self.websocket = websocket
                    self.logger.info(f"Websocket connected to {websocket_url}")
                    retry_count = 0  # Reset retry count on successful connection

                    while self.running:
                        data = await websocket.recv()
                        # Only add to queue if there's room and not muted
                        if not self.mute_input and self.input_queue.qsize() < self.config.MAX_QUEUE_SIZE:
                            self.input_queue.put(data)
                            if self.config.DEBUG and random.random() < 0.01:
                                self.logger.debug(f"Websocket audio chunk received: {len(data)} bytes")
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Websocket error (attempt {retry_count}/{max_retries}): {e}")
                if self.config.DEBUG:
                    self.logger.debug(traceback.format_exc())
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff

        if retry_count >= max_retries:
            self.logger.error(f"Websocket connection failed after {max_retries} attempts")
        self.running = False

    def set_mute(self, mute_state):
        """Mute or unmute the audio input"""
        if self.mute_input != mute_state:
            self.mute_input = mute_state
            self.logger.debug(f"Audio input mute set to: {mute_state}")

    async def stop(self):
        """Stop audio input and clean up resources"""
        self.logger.info("Stopping audio input")
        self.running = False

        try:
            if self.input_mode == "microphone" and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.logger.debug("Microphone stream closed")

            if self.audio_interface:
                self.audio_interface.terminate()
                self.logger.debug("PyAudio interface terminated")

            self.websocket = None
            self.logger.info("Audio input stopped")
        except Exception as e:
            self.logger.error(f"Error stopping audio input: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())


# Voice Activity Detection Module
class VADModule:
    def __init__(self, config):
        self.config = config
        self.vad = __import__("webrtcvad").Vad(config.VAD_AGGRESSIVENESS)
        self.speech_active = False
        self.last_active_time = 0
        self.last_speech_frame_time = 0
        self.silent_frames = 0
        self.speech_frames = 0
        self.buffer = b""
        self.energy_history = []
        self.speech_segments = []
        self.logger = logging.getLogger("SpeechSystem.VAD")

        # Add sliding window tracking
        self.sliding_windows = []
        self.window_size_bytes = config.get_window_size_bytes()
        self.window_overlap_bytes = config.get_window_overlap_bytes()
        self.total_speech_bytes = 0

        # Enhanced configuration for better speech detection
        self.min_speech_frames = 3  # Minimum frames to consider as speech start
        self.decision_frames = 5  # Number of frames to base decisions on
        self.energy_threshold = config.ENERGY_THRESHOLD  # Energy threshold for speech
        self.speech_timeout = 1.4  # Maximum time without speech before considering end

        # Initialize context buffer
        self.prev_frames = []

        # Add intelligent end detection features
        self.current_transcription = ""  # Track transcription for syntax analysis
        self.extended_silence_threshold = config.SILENCE_THRESHOLD * 2  # Double timeout
        self.intelligent_end_enabled = True  # Enable feature

        self.logger.info(f"VAD initialized with aggressiveness {config.VAD_AGGRESSIVENESS}")
        if self.config.DEBUG:
            self.logger.debug(f"VAD parameters: min_speech_frames={self.min_speech_frames}, "
                              f"decision_frames={self.decision_frames}, "
                              f"energy_threshold={self.energy_threshold}, "
                              f"speech_timeout={self.speech_timeout}, "
                              f"extended_silence={self.extended_silence_threshold}")

    def process_frame(self, frame):
        """Process a single frame of audio and detect speech with improved algorithm"""
        if frame is None or len(frame) == 0:
            self.logger.warning("Received empty frame")
            return "inactive", None

        try:
            frame_size = len(frame)
            is_speech = False
            current_time = time.time()

            # Extract energy for better decision making
            audio_data = np.frombuffer(frame, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            self.energy_history.append(rms)

            # Keep history limited
            if len(self.energy_history) > self.decision_frames:
                self.energy_history.pop(0)

            # Calculate average energy
            avg_energy = np.mean(self.energy_history) if self.energy_history else 0

            # Use WebRTC VAD for reliable speech detection
            if frame_size == 320 or frame_size == 640:  # 20ms or 40ms at 16kHz
                try:
                    vad_result = self.vad.is_speech(frame, self.config.SAMPLE_RATE)
                    # Combine VAD with energy for more robust detection
                    is_speech = vad_result and (avg_energy > self.energy_threshold)
                except Exception as vad_error:
                    self.logger.warning(f"VAD processing error: {vad_error}")
                    # If VAD fails, use energy-based detection
                    is_speech = avg_energy > self.energy_threshold
            else:
                # Frame size is not 20ms or 40ms, use energy-based detection
                is_speech = avg_energy > self.energy_threshold
                if self.config.DEBUG and random.random() < 0.01:
                    self.logger.debug(
                        f"Using energy-based detection for frame size {frame_size}: energy={avg_energy}, is_speech={is_speech}")

            if self.speech_active and self.config.SLIDING_WINDOW_ENABLED:
                # Add frame to buffer
                if len(self.buffer) < self.config.MAX_BUFFER_SIZE:
                    self.buffer += frame
                    self.total_speech_bytes += len(frame)

                # Check if we need to create a sliding window
                if len(self.buffer) >= self.window_size_bytes:
                    # Create a sliding window and keep overlap for continuation
                    self.sliding_windows.append(self.buffer)
                    self.logger.debug(f"Created sliding window #{len(self.sliding_windows)}: {len(self.buffer)} bytes")

                    # Keep overlap portion for next window
                    self.buffer = self.buffer[-self.window_overlap_bytes:] if len(
                        self.buffer) >= self.window_overlap_bytes else b""

            # State machine for more reliable speech detection
            if is_speech:
                self.speech_frames += 1
                self.silent_frames = 0
                self.last_speech_frame_time = current_time
                # Potential end of speech, add to buffer during pause
                if len(self.buffer) < self.config.MAX_BUFFER_SIZE:
                    self.buffer += frame

                if not self.speech_active and self.speech_frames >= self.min_speech_frames:
                    # More reliable speech start detection
                    self.speech_active = True
                    b"".join(self.prev_frames) + frame  # Include previous frames for context
                    if self.config.DEBUG:
                        self.logger.debug(f"Speech start detected: frames={self.speech_frames}, energy={avg_energy}")
                    return "start", None
                elif self.speech_active:
                    # Ongoing speech, add to buffer
                    if len(self.buffer) < self.config.MAX_BUFFER_SIZE:
                        self.buffer += frame
                    self.last_active_time = current_time
                    return "active", None
                else:
                    # Collecting frames leading to speech
                    if len(self.prev_frames) >= self.min_speech_frames:
                        self.prev_frames.pop(0)
                    self.prev_frames.append(frame)
                    return "inactive", None
            else:
                self.speech_frames = 0
                self.silent_frames += 1

                if self.speech_active:
                    # Potential end of speech, add to buffer during pause
                    if len(self.buffer) < self.config.MAX_BUFFER_SIZE:
                        self.buffer += frame

                    # Calculate silence duration since last speech
                    silence_duration = current_time - self.last_speech_frame_time

                    # INTELLIGENT END DETECTION - Check multiple criteria

                    # 1. Check standard silence threshold
                    if silence_duration > self.config.SILENCE_THRESHOLD:
                        # Standard silence-based end detection
                        self.speech_active = False
                        segment = self.get_combined_audio()
                        buffer_length = len(segment)
                        self.buffer = b""
                        self.sliding_windows = []
                        self.total_speech_bytes = 0
                        if self.config.DEBUG:
                            self.logger.debug(
                                f"Speech end detected (standard silence): silence={silence_duration:.2f}s, buffer={buffer_length} bytes")
                        return "end", segment

                    # 2. Check for intelligent syntax-based end detection with shorter silence
                    elif self.intelligent_end_enabled and silence_duration > (self.config.SILENCE_THRESHOLD * 0.8):
                        # Apply if transcription ends with sentence-ending punctuation
                        if self.current_transcription and any(
                            self.current_transcription.strip().endswith(p) for p in ['.', '!', '?']):
                            self.logger.info(
                                f"Intelligent end detection triggered: '{self.current_transcription[-15:] if len(self.current_transcription) > 15 else self.current_transcription}' (silence: {silence_duration:.2f}s)")

                            # End speech segment based on syntax
                            self.speech_active = False
                            segment = self.get_combined_audio()
                            buffer_length = len(segment)
                            self.buffer = b""
                            self.sliding_windows = []
                            self.total_speech_bytes = 0

                            return "end", segment

                    # 3. Check for extended silence (2x normal threshold)
                    elif silence_duration > self.extended_silence_threshold:
                        # Extended timeout reached - definitely end
                        self.speech_active = False
                        segment = self.get_combined_audio()
                        buffer_length = len(segment)
                        self.buffer = b""
                        self.sliding_windows = []
                        self.total_speech_bytes = 0
                        if self.config.DEBUG:
                            self.logger.debug(
                                f"Speech end detected (extended timeout): silence={silence_duration:.2f}s, buffer={buffer_length} bytes")
                        return "end", segment

                    elif silence_duration > 0.1:  # Short pause detection
                        if self.config.DEBUG and random.random() < 0.05:
                            self.logger.debug(f"Thinking pause detected: {silence_duration:.2f}s")
                        return "thinking", None
                    else:
                        return "active", None  # Still consider as active if pause is very short
                else:
                    # No speech, maintain rolling context buffer
                    if len(self.prev_frames) >= self.min_speech_frames:
                        self.prev_frames.pop(0)
                    self.prev_frames.append(frame)
                    return "inactive", None
        except Exception as e:
            self.logger.error(f"Error in VAD processing: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())
            return "error", None
        return "inactive", None

    def update_current_transcription(self, transcription):
        """Update the current transcription for intelligent end detection"""
        self.current_transcription = transcription
        if self.config.DEBUG and transcription and len(transcription) > 10:
            self.logger.debug(f"Updated transcription for VAD: '{transcription[:40]}...'")

    # Keep existing methods
    def get_combined_audio(self):
        """Combine all windows and current buffer into a single audio segment"""
        if not self.sliding_windows:
            return self.buffer

        result = b""
        for i, window in enumerate(self.sliding_windows):
            if i == 0:
                # Add first window completely
                result += window
            else:
                # Skip overlap from previous window
                overlap = min(self.window_overlap_bytes, len(window))
                result += window[overlap:]

        # Add final buffer (if not empty)
        if self.buffer:
            # Skip overlap with the last window if it exists
            if self.sliding_windows:
                overlap = min(self.window_overlap_bytes, len(self.buffer))
                result += self.buffer[overlap:]
            else:
                result += self.buffer

        self.logger.debug(f"Combined {len(self.sliding_windows)} windows + buffer into {len(result)} bytes")
        return result

    def reset(self):
        """Reset the VAD state including sliding windows"""
        self.speech_active = False
        self.buffer = b""
        self.silent_frames = 0
        self.speech_frames = 0
        self.energy_history = []
        self.prev_frames = []
        self.sliding_windows = []
        self.total_speech_bytes = 0
        self.current_transcription = ""  # Reset transcription buffer
        self.logger.debug("VAD state reset")


# Speech Deserializer Module
class SpeechDeserializerModule:
    def __init__(self, config):
        self.config = config
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.live_update_buffer = None
        self.last_transcription = ""
        self.processing_lock = threading.Lock()
        self.logger = logging.getLogger("SpeechSystem.SpeechDeserializer")

        # Start worker thread for non-blocking processing
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        self.models = ['whisper-large-v3', 'whisper-large-v3-turbo']
        self.model_index = 0

        # ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Add window tracking for concatenation
        self.transcription_segments = {}
        self.segment_counter = 0
        self.last_tts_output = ""
        self.tts_prefix = "USER: "
        self.logger.info("Speech deserializer initialized")

    def _worker(self):
        """Worker thread that processes tasks from the queue"""
        # Existing worker code
        while True:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Sentinel to stop the thread
                    break

                # Unpack the task with segment ID
                if len(task) == 4:
                    audio_data, is_final, callback, segment_id = task
                else:
                    # Backward compatibility for older task format
                    audio_data, is_final, callback = task
                    segment_id = "legacy"

                try:
                    self.logger.debug(
                        f"Processing audio: length={len(audio_data)}, is_final={is_final}, segment={segment_id}")

                    def helper():
                        try:
                            # Transcribe the audio
                            result = self._transcribe(audio_data)
                            with self.processing_lock:

                                # Store segment result if needed for later concatenation
                                if segment_id != "final" and segment_id != "legacy":
                                    self.transcription_segments[segment_id] = result
                                    self.logger.debug(f"Stored transcription segment {segment_id}: {result[:50]}...")

                                # For final segments, try to concatenate previous segments if available
                                if is_final and self.transcription_segments:
                                    combined_result = self._concatenate_transcriptions(result)
                                    self.logger.info(
                                        f"Created concatenated transcription from {len(self.transcription_segments) + 1} segments")
                                    result = combined_result

                                    # Clear segments after concatenation
                                    self.transcription_segments = {}

                                # Store for reference
                                if is_final:
                                    self.last_transcription = result

                                # Queue result for callback processing
                                if callback:
                                    self.results_queue.put((result, is_final, callback))
                        finally:
                            self.task_queue.task_done()

                    threading.Thread(target=helper, daemon=True).start()

                except Exception as e:
                    self.logger.error(f"Transcription processing error: {e}")
                    if self.config.DEBUG:
                        self.logger.debug(traceback.format_exc())

            except queue.Empty:
                # Timeout on queue.get - just continue
                pass
            except Exception as e:
                self.logger.error(f"Speech deserializer worker error: {e}")
                if self.config.DEBUG:
                    self.logger.debug(traceback.format_exc())
                time.sleep(0.5)  # Prevent rapid error loops

    def _concatenate_transcriptions(self, final_segment_text=""):
        """
        Intelligently concatenate transcription segments into a complete, accurate transcription.

        This method analyzes chronological progression of speech segments and identifies
        key phrases to construct a coherent transcription.

        Args:
            final_segment_text: Text from the final segment (if applicable)

        Returns:
            Complete, clean transcription string
        """
        if not self.transcription_segments and not final_segment_text:
            return ""

        # Prepare final text if provided
        final_text = final_segment_text.strip() if final_segment_text else ""

        # Extract segments with timestamp information for chronological analysis
        segments = []

        for key, text in self.transcription_segments.items():
            text = text.strip()
            if not text:
                continue

            # Extract timestamp for live segments (for chronological ordering)
            timestamp = float('inf')  # Default high value
            if '_live_' in key and len(key.split('_')) >= 4:
                with contextlib.suppress(ValueError):
                    timestamp = int(key.split('_')[-1])
            elif '_window_' in key:
                # Window segments come after live segments but before final
                timestamp = float('inf') - 100

            segments.append({
                'key': key,
                'text': text,
                'timestamp': timestamp,
                'length': len(text)
            })

        # Add final segment if provided
        if final_text:
            segments.append({
                'key': 'final',
                'text': final_text,
                'timestamp': float('inf'),
                'length': len(final_text)
            })

        # Sort segments chronologically
        segments.sort(key=lambda x: x['timestamp'])

        # Identify distinct phrases by analyzing segment content progression
        phrases = []
        current_phrase = None

        for segment in segments:
            if not current_phrase:
                # Start first phrase
                current_phrase = {
                    'text': segment['text'],
                    'segments': [segment],
                    'words': set(segment['text'].lower().split())
                }
                continue

            # Check if this segment is a continuation/refinement of current phrase
            # or if it introduces a new phrase

            # Extract words for comparison
            segment_words = set(segment['text'].lower().split())

            # Compute similarity (shared words divided by smaller set size)
            shared_words = len(current_phrase['words'].intersection(segment_words))
            similarity = shared_words / min(len(current_phrase['words']), len(segment_words)) if min(
                len(current_phrase['words']), len(segment_words)) > 0 else 0

            # Check if current segment is longer version of current phrase
            is_extension = current_phrase['text'].lower() in segment['text'].lower()

            if similarity > 0.5 or is_extension:
                # This segment refines/extends the current phrase
                current_phrase['segments'].append(segment)

                # If this segment is longer, update phrase text
                if len(segment['text']) > len(current_phrase['text']):
                    current_phrase['text'] = segment['text']
                    current_phrase['words'] = segment_words
            else:
                # This segment introduces a new phrase
                phrases.append(current_phrase)
                current_phrase = {
                    'text': segment['text'],
                    'segments': [segment],
                    'words': segment_words
                }

        # Don't forget to add the last phrase
        if current_phrase:
            phrases.append(current_phrase)

        # For each phrase, select the best representative segment (usually the longest)
        best_segments = []

        for phrase in phrases:
            # Choose longest segment as best representative
            best = max(phrase['segments'], key=lambda x: x['length'])
            best_segments.append(best)

        # Combine best segments into final transcription
        final_transcription = ""

        for i, segment in enumerate(best_segments):
            if i == 0:
                final_transcription = segment['text']
            else:
                # Check for overlap with previous text to avoid repetition
                overlap = self._find_text_overlap(final_transcription.lower(), segment['text'].lower())

                if overlap > 5:  # Significant overlap
                    # Add only the non-overlapping part
                    final_transcription += segment['text'][overlap:]
                else:
                    # No significant overlap, just append with a space
                    final_transcription += " " + segment['text']

        # Clean up the final text
        final_transcription = " ".join(final_transcription.split())

        # Filter out TTS echo
        final_transcription = self._filter_tts_echo(final_transcription)

        self.logger.info(f"Created concatenated transcription from {len(best_segments)} distinct phrases")

        # Clear segments after processing
        self.transcription_segments = {}

        return final_transcription

    def _filter_tts_echo(self, text):
        """
        Filter out TTS echo from transcription
        """
        # Check if the transcription contains our TTS prefix
        if self.tts_prefix.lower() in text.lower():
            self.logger.warning(f"Detected TTS echo in transcription: '{text}'")
            return ""

        # Check if the transcription is very similar to our last TTS output
        if self.last_tts_output and text:
            # Compare with last TTS output without the prefix
            actual_output = self.last_tts_output.replace(self.tts_prefix, "").strip().lower()
            if actual_output and (actual_output in text.lower() or text.lower() in actual_output):
                self.logger.warning(f"Detected TTS echo: '{text}' matches '{actual_output}'")
                return ""

        return text

    def set_last_tts_output(self, text):
        """Record the last TTS output to filter echoes"""
        self.last_tts_output = text
        self.logger.debug(f"Recorded TTS output: '{text}'")

    @staticmethod
    def _find_text_overlap(text1, text2, min_chars=5):
        """
        Find the number of overlapping characters between the end of text1
        and the beginning of text2
        """
        # Simple but effective overlap detection
        max_overlap = min(len(text1), len(text2), 30)  # Limit to 30 chars max

        for overlap_size in range(max_overlap, min_chars - 1, -1):
            if text1[-overlap_size:].lower() == text2[:overlap_size].lower():
                return overlap_size

        return 0

    def _transcribe(self, audio_data):
        """Transcribe audio data to text using Groq API"""
        temp_wav = None
        try:
            # Only create temporary files for significant audio data
            if len(audio_data) < 4000:  # Skip very small audio chunks
                self.logger.debug(f"Skipping small audio segment: {len(audio_data)} bytes")
                return ""
            # Create a temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            wav_path = temp_wav.name
            temp_wav.close()  # Close the file handle immediately

            # Write the WAV file
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(self.config.CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.config.SAMPLE_RATE)
                wav_file.writeframes(audio_data)

            self.logger.debug(f"Created temporary WAV file: {wav_path}, size={len(audio_data)} bytes")

            # Make API call to Groq for transcription
            with open(wav_path, "rb") as file:
                self.logger.debug("Sending transcription request to Groq API")
                start_time = time.time()
                response = self.client.audio.transcriptions.create(
                    file=file,
                    model=self.models[self.model_index],
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
                duration = time.time() - start_time
                if duration > 2:
                    self.model_index = 0 if self.model_index else 1
                self.logger.debug(f"Groq API response received in {duration:.2f}s")
                return response.text

        except Exception as e:
            self.logger.error(f"Transcription API error: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())

        finally:
            # Clean up temporary file
            if temp_wav and os.path.exists(temp_wav.name):
                try:
                    os.unlink(temp_wav.name)
                    self.logger.debug(f"Temporary file removed: {temp_wav.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {temp_wav.name}: {e}")
        return ""

    def process_audio(self, audio_data, is_final=False, callback=None, segment_id=None):
        """Queue audio data for transcription with segment identification"""
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("Received empty audio data for processing")
            return

        # If segment_id is not provided, generate one
        if segment_id is None:
            if is_final:
                segment_id = "final"  # Special marker for final segment
            else:
                self.segment_counter += 1
                segment_id = f"segment_{self.segment_counter}"

        # Queue for processing
        self.task_queue.put((audio_data, is_final, callback, segment_id))
        self.logger.debug(
            f"Audio queued for processing: length={len(audio_data)}, is_final={is_final}, segment_id={segment_id}")

    async def process_results(self):
        """Process transcription results from the queue"""
        try:
            results_processed = 0
            while not self.results_queue.empty():
                result, is_final, callback = self.results_queue.get()
                if callback:
                    try:
                        callback(result, is_final)
                        results_processed += 1
                    except Exception as e:
                        self.logger.error(f"Error in transcription callback: {e}")
                        if self.config.DEBUG:
                            self.logger.debug(traceback.format_exc())
                self.results_queue.task_done()

            if results_processed > 0 and self.config.DEBUG:
                self.logger.debug(f"Processed {results_processed} transcription results")

        except Exception as e:
            self.logger.error(f"Error processing transcription results: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())

    def shutdown(self):
        """Clean up resources"""
        self.logger.info("Shutting down speech deserializer")
        self.task_queue.put(None)  # Signal worker thread to stop
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            self.logger.warning("Speech deserializer worker thread did not terminate")
        self.executor.shutdown(wait=False)
        self.logger.debug("Speech deserializer shutdown complete")


def get_game_email(credentials_path=r"C:\Users\Markin\Workspace\ToolBoxV2\client_secret.json"):
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow

    def init_services():
        """
        Initialize Gmail and Calendar services
        """
        from googleapiclient.discovery import build

        gmail_service = build('gmail', 'v1', credentials=credentials)
        calendar_service = build('calendar', 'v3', credentials=credentials)
        return gmail_service, calendar_service

    try:
        credentials = Credentials.from_authorized_user_file('token/google_token.json')
        if credentials.valid:
            return init_services()
    except FileNotFoundError:
        pass

    flow = Flow.from_client_secrets_file(
        credentials_path,
        scopes=[
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/calendar'
        ],
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # Use 'urn:ietf:wg:oauth:2.0:oob' for desktop apps
    )

    # Generate the authorization URL
    authorization_url, _ = flow.authorization_url(
        access_type='offline',  # Allows obtaining refresh token
        prompt='consent'  # Ensures user is always prompted for consent
    )

    print(authorization_url)
    import webbrowser
    webbrowser.open(authorization_url)

    authorization_code = input("Log In code:")
    flow.fetch_token(code=authorization_code)
    credentials = flow.credentials

    """
    Save the obtained credentials to a file for future use
    """
    if not os.path.exists('token'):
        os.makedirs('token')

    with open('token/google_token.json', 'w') as token_file:
        token_file.write(credentials.to_json())

    return init_services()


class TTSModule:
    def __init__(self, config, on_tts_start, on_tts_end, vad_name="ISAA0"):
        #from RealtimeTTS import (
        #    KokoroEngine,
        #    SystemEngine,
        #    TextToAudioStream,
        #)
        self.config = config
        self.on_tts_start = on_tts_start
        self.on_tts_end = on_tts_end
        self.logger = logging.getLogger("SpeechSystem.TTS")
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.current_text = None
        self.can_be_interrupted = True
        self.stream = None  # Will be instantiated per task based on language
        self.ecco = None
        self.wake_words = ["computer", "system", "isa", "isaa", "issa", "iza", "pc", "agent", "i", "ich"]

        # Start worker thread for non-blocking TTS
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        self.de_engine = SystemEngine("Hedda")

        self.logger.info("Initializing Kokoro engine for TTS")
        self.en_engine = KokoroEngine(default_lang_code="a", default_voice="af_heart")

        self.stream = TextToAudioStream(
            engine=self.en_engine,
            # on_text_stream_start=self._on_text_stream_start,
            # on_text_stream_stop=self._on_text_stream_stop,
            on_audio_stream_start=self._on_audio_stream_start,
            on_audio_stream_stop=self._on_audio_stream_stop,
            # on_character=self._on_character,
            muted=getattr(self.config, "MUTED", False),
            # level=logging.DEBUG if getattr(self.config, "DEBUG", False) else logging.WARNING
        )

        from toolboxv2.mods.isaa.CodingAgent.live import Pipeline
        from toolboxv2.mods.isaa.extras.session import ChatSession
        from toolboxv2.mods.isaa.isaa_modi import browse_website
        from toolboxv2.mods.isaa.subtools.file_loder import route_local_file_to_function

        if not get_app("vad.TTS").mod_online("isaa"):
            get_app("vad.TTS").get_mod("isaa").init_isaa(build=True)
        if get_app("vad.TTS").mod_online("isaa"):
            get_app("vad.TTS").get_mod("isaa").init_isaa(build=True)
            self.agent0 = get_app("vad.TTS").get_mod("isaa").get_agent(vad_name,
                                                                       model="openrouter/google/gemini-2.0-flash-lite-001")
            # self.agent0.world_model = get_app("vad.TTS").get_mod("isaa").get_agent("self").world_model
            self.chat_session = ChatSession(self.agent0.memory,
                                            space_name=f"ChatSession/{self.agent0.amd.name}/Pipeline.session",
                                            max_length=200)
            # variables: List = [] #self.agent1.functions.copy()
            i = get_app("vad.TTS").get_mod("isaa")
            gmail_service, calendar_service = get_game_email()

            def memory_search(query: str):
                ress = i.get_memory().query(query, to_str=True)

                if not ress:
                    return "no informations found for :" + query

                return ress

            def path_to_context_list(path: str):
                """Local path file or folder to list of str ( Context )"""
                loder, docs_loder = route_local_file_to_function(path)
                docs = docs_loder()
                return [doc.page_content for doc in docs]

            def browse_website_wit_question(url: str, question: str):
                return browse_website(url, question, i.mas_text_summaries)


            def get_clipboard():
                """
                Retrieve text data from the system clipboard.

                Returns:
                str: Text content currently stored in the clipboard
                """
                try:
                    clipboard_content = __import__("pyperclip").paste()
                    return clipboard_content
                except Exception as e:
                    print(f"Error retrieving clipboard data: {e}")
                    return None

            def to_clipboard(data):
                """
                Save text data to the system clipboard.

                Args:
                data (str): Text content to be saved to the clipboard

                Returns:
                bool: True if successful, False otherwise
                """
                try:
                    pyperclip.copy(str(data))
                    return True
                except Exception as e:
                    print(f"Error saving data to clipboard: {e}")
                    return False

            variables = {
                "get_clipboard": get_clipboard,
                "to_clipboard": to_clipboard,
                "memory_search": memory_search,
                "shell_tool_function": i.shell_tool_function,
                "a_web_search": i.web_search,
                "mas_text_summaries": i.mas_text_summaries,
                "run_agent": i.run_agent,
                "get_agent": i.get_agent,
                "create_task_chain": i.create_task_chain,
                "crate_and_run_task": i.crun_task,
                "run_task": i.run_task,
                "get_task": i.get_task,
                "load_task": i.load_task,
                "save_task": i.save_task,
                "remove_task": i.remove_task,
                "list_task": i.list_task,
                "gmail_service": gmail_service,
                "calendar_service": calendar_service,
                "browse_website": browse_website_wit_question,
                "path_to_context_list": path_to_context_list
            }
            self.pipe: Pipeline = get_app("vad.TTS").get_mod("isaa").get_pipe(
                get_app("vad.TTS").get_mod("isaa").get_agent(f"VadAgent-{self.agent0.amd.name}-o3", model="openrouter/openai/o3-mini"),
                verbose=True, timeout_timer=1, variables=variables, max_iter=4)
            self.agent0.stream = False

        def on_press(key):
            """
            Callback that is called whenever any key is pressed.
            Sets the asyncio event to signal a key press.
            """
            self.stream.stop()
            self.interrupt()
            # Optionally, you could log or handle 'key' here.

        # Start a keyboard listener that works on all platforms.
        keyboard = __import__("pynput.keyboard")
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.logger.info("TTS module (RealtimeTTS) initialized")

    def _detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            self.logger.debug(f"Detected language: {lang}")
            return lang
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "en"  # Fallback to English if detection fails

    def _init_stream(self, lang: str):
        """
        Initialize the TextToAudioStream with the proper engine and callbacks.
        For English, use the KokoroEngine with two voices;
        for German, use the best available GermanEngine.
        """
        if lang.startswith("de"):
            self.logger.info("Loading German engine for TTS")
            self.stream.load_engine(self.de_engine)
        else:
            self.logger.info("Loading Kokoro engine for TTS")
            self.stream.load_engine(self.en_engine)

    # Callback methods for TextToAudioStream events
    def _on_text_stream_start(self):
        self.logger.debug("Text stream started")

    def _on_text_stream_stop(self):
        self.logger.debug("Text stream stopped")

    def _on_audio_stream_start(self):
        self.logger.debug("Audio stream started")
        self.is_speaking = True
        if self.on_tts_start:
            self.on_tts_start()

    def _on_audio_stream_stop(self):
        self.logger.debug("Audio stream stopped")
        if self.on_tts_end:
            self.on_tts_end(self.current_text)

    def _on_character(self, char):
        # Optionally log each character as it’s processed
        self.logger.debug(f"Processing character: {char}")

    async def _transmutate(self, text, talk):
        process = False
        text_l = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_l:
                process = True
                break
        if not process:
            return
        if self.ecco:
            return await talk(text)
        if text == self.agent0.user_input:
            return ""
        if self.agent0.last_result is not None and text in self.agent0.last_result:
            return ""
        text = text[5:]

        # res = await self.agent0.a_mini_task(text, task_from="user", mini_task=prompts, persist=False)

        class UserResponse(BaseModel):
            analysis_and_planning: str = Field(...,
                                               description="""1. Analyze input type (chat vs technical task)
        2. Identify required capabilities from these options:
           - Code execution (Python)
           - automation
           - File system operations
           - Variable management
           - Session state control
        3. Determine delegation strategy:
           a) Direct response for conversational/trivial requests
           b) Inner execution for tasks requiring:
              * Code evaluation/modification
              * Complex calculations
              * Multi-step workflows
        4. If delegation needed:
           - Formulate precise technical task description
           - Estimate required iterations (1-25)
           - Plan error recovery strategy
        5. Prepare fallback options for execution failures""")

            inner: str | None = Field(...,
                                         description="""Technical task specification for Pipeline including:
        - Language context (Python)
        - Required variables/state
        - Expected outputs
        - Error handling approach
        - Resource requirements
        Examples:
        1. 'Calculate fibonacci sequence up to n=100'
        2. 'Scrape product data from example.com using a_web_search("example.com")'
        3. 'Modify ML model parameters and retrain'""")

            inner_max_iterations: int | None = Field(...,
                                                        description="""Complexity estimation:
        1-3: Simple code execution/query
        4-8: Multi-step analysis
        9-12: Complex workflow with state management
        13-25: Advanced problem solving with iterations""")

            response: str = Field(...,
                                  description="""User-facing response following these rules:
        1. For delegated tasks:
           - Acknowledge request
           - Explain technical approach in simple terms
           - Provide progress updates
           - Summarize final results
        2. Include interactive elements:
           - Option to view execution details
           - Alternative approaches
           - Parameter adjustments
        3. Format technical outputs as:
           - Key metrics first
           - Visualizations when applicable
           - Error summaries with solutions
        4. Maintain conversational flow between system and user""")

        history = await self.chat_session.get_reference(text, to_str=True)
        agent_res = await self.agent0.a_format_class(
            UserResponse,text,
            message=[{'role': 'system', 'content':
            f"""You are Isaa an AI assistant designed to provide insightful, direct, and engaging responses while maintaining the ability to handle complex tasks when necessary. Your primary goal is to uncover the truth and engage in deep, analytical discussions on various topics. You have a tendency to go into great detail and are highly intelligent, but your social skills are underdeveloped. You are matter-of-fact and may not always recognize if your statements could be perceived as offensive. Your responses should be based on logical arguments rather than emotional ones.
The provide input comes from stt so their might be errors and or duplication ignor them!

Follow these steps:

1. Analyze the input to determine if it's a conversational request, a simple task, or a complex task requiring delegation.
2. Based on your analysis, choose one of the following actions:
   a) For conversational inputs: Respond directly with an insightful, detailed, and engaging message.
   b) For simple tasks: Use the inner system to delegate the task to a more capable system and provide an initial response.

3. Maintain an analytical tone and respond to the user, even when delegating tasks.
4. Provide detailed responses that thoroughly explore the topic at hand. There is no strict limit on response length, but ensure your answers are structured and easy to follow.
5. If the user's request is unclear or lacks necessary details, ask clarification questions before proceeding.
6. When delegating tasks, ensure that the request to the inner call is precise and comprehensive. as well the immediate response to the user.
7. Always look at the information fom the WORLD MODEL most of the time its helpful for context and guidance.
If you need to delegate a task or perform any non-conversational action, use the inner
Remember, the response section is for user interaction. Any task that is not conversational must be delegated to the inner system
Here is the word model you should use:
You are Isaa, an AI orchestrator managing these capabilities:

**Pipeline Core Features**
1. Code Execution Engine:
   - Python with full async support
   - Real-time output streaming
   - Automatic dependency handling

2. State Management:
   - Variable versioning
   - Session snapshots
   - Cross-execution context
   - Automatic state recovery

3. Analysis Tools:
   - Execution tracing
   - Performance profiling
   - Error diagnostics
   - Memory monitoring

**Decision Framework**
- Direct Response When:
  1. Conceptual questions
  2. Simple calculations
  3. Non-technical discussions

- Delegate to Inner System When:
  1. Requires code execution
  2. Needs browser interaction
  3. Involves complex data processing
  4. Requires persistent state management

**Execution Protocol**
1. For delegated tasks:
   - Set clear success criteria
   - Define resource boundaries
   - Specify output format
   - Plan rollback strategy

2. Error Handling:
   - Automatic retry with backoff
   - Alternative implementations
   - Partial result preservation
   - Clean failure states

**Current Environment**
{self.agent0.show_world_model()}
{self.pipe.show_vars()}

**Interaction Guidelines**
1. Maintain chain-of-thought visibility
2. Balance technical depth with accessibility
3. Provide execution previews
4. Offer optimization suggestions
Extra history context:
{history}
Now, you're ready to engage in deep, analytical discussions with users!"""}]+self.chat_session.get_past_x(160).copy()
        )
        it = 0
        while it <= 4:
            it += 1
            aap = agent_res.get('analysis_and_planning', 'N/A')
            if aap: aap = aap.replace('.', '.\n ')
            inn = agent_res.get('inner', 'N/A')
            if inn: inn = inn.replace('.', '.\n ')
            resp = agent_res.get('response', 'N/A')
            if resp: resp = resp.replace('.', '.\n ')
            self.pipe.verbose_output.formatter.print_section(
            "Step Result",
            f"analysis_and_planning: {aap}\n"
            f"inner: {inn}\n"
            f"inner_max_iterations: {agent_res.get('inner_max_iterations', 'None')}\n"
            f"response: {resp}\n"
            )

            await talk(agent_res.get('response'))
            if it == 1:
                await self.chat_session.add_message({'role': 'user', 'content': text})
            await self.chat_session.add_message({'role': 'assistant', 'content': agent_res.get('response')})
            _res = ''

            if agent_res.get('inner') is None:
                break

            if it <= 4:

                self.pipe.ipython.user_ns["world_model"] = self.agent0.world_model.copy()
                self.pipe.max_iter = min(max(1, int(agent_res.get('inner_max_iterations', '4'))), 25)
                pipe_res = await self.pipe.run(agent_res.get('inner'), True)
                _res = ""
                for i, exr in enumerate(pipe_res.execution_history):
                    _res += f"Step {i}:\n result -> {exr.result}" + (
                        "\n" if exr.error is None else "Error:" + str(exr.error) + '\n')
                _res += pipe_res.result

                await self.chat_session.add_message({'role': 'assistant', 'content': "Inner Agent report:\n" + _res})

                agent_res = await self.agent0.a_format_class(
                    UserResponse, "Response to the user focus on the chat history (last inner agent assistant massage)! , optionally on an error run an inner command with an dirent approach!",
                    message= self.chat_session.get_past_x(
                        6*4).copy()
                )

            await self.agent0.flow_world_model(text + '\nResult:' + agent_res.get('response') + _res)


    def _worker(self):
        """Worker thread processing queued TTS tasks using RealtimeTTS."""
        self.logger.debug("TTS worker thread started")
        lock = threading.Lock()

        async def __awork():
            las_lang = ["en"]
            while True:
                try:
                    task = self.tts_queue.get(timeout=0.5)
                    if task is None:  # Sentinel value for shutdown
                        self.logger.debug("Worker received shutdown signal")
                        break

                    text, callback_start, callback_end = task

                    async def talk(text):
                        if not text:
                            return self.is_speaking

                        self.current_text = text

                        # Auto language detection
                        lang = self._detect_language(text)
                        if las_lang[0] != lang:
                            self._init_stream(lang)
                        las_lang[0] = lang

                        # Trigger any provided callback before starting
                        if callback_start:
                            callback_start()

                        self.logger.debug(f"Starting TTS for text: {text[:50]}{'...' if len(text) > 50 else ''}")
                        try:
                            # Start streaming the text into audio.
                            # The callbacks _on_text_stream_start and _on_text_stream_stop will trigger
                            # on_tts_start and on_tts_end respectively.
                            self.stream.feed([text]).play(log_synthesized_text=True)
                        except Exception as e:
                            self.logger.error(f"TTS streaming error: {e}")
                            if getattr(self.config, "DEBUG", False):
                                self.logger.debug(traceback.format_exc())
                        finally:
                            self.is_speaking = False
                            #self.current_text = None
                            if callback_end:
                                callback_end(text)
                            #self.tts_queue.task_done()
                            pass
                        return not self.is_speaking

                    def helper_talk_(text_):
                        # Create an asyncio Event that will be triggered on key press.
                        with lock:
                            asyncio.run(talk(text_))

                    async def helper_talk(text_):
                        threading.Thread(target=helper_talk_, args=(text_,), daemon=True).start()

                    self.is_speaking = True
                    await self._transmutate(text, helper_talk)


                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"TTS worker error: {e}")
                    if getattr(self.config, "DEBUG", False):
                        self.logger.debug(traceback.format_exc())
                    time.sleep(0.1)
                finally:
                    self.is_speaking = False
                    self.current_text = None
                    if self.tts_queue.unfinished_tasks >= 1:
                        self.tts_queue.task_done()

        asyncio.run(__awork())

    def speak(self, text, callback_start=None, callback_end=None):
        """
        Queue text for TTS conversion. Clears any pending texts before queuing the new text.
        """
        try:
            cleared_items = 0
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                    cleared_items += 1
                except queue.Empty:
                    break
            if cleared_items > 0:
                self.logger.debug(f"Cleared {cleared_items} pending TTS items from queue")

            self.tts_queue.put((text, callback_start, callback_end))
            self.logger.debug(f"Queued text for TTS: {text[:50]}{'...' if len(text) > 50 else ''}")
        except Exception as e:
            self.logger.error(f"Error queuing text for TTS: {e}")
            if getattr(self.config, "DEBUG", False):
                self.logger.debug(traceback.format_exc())

    def interrupt(self):
        """
        Interrupt the current speech if speaking. Attempts to call the stream's stop function.
        """
        if self.is_speaking and self.can_be_interrupted:
            self.logger.info("Interrupting current TTS speech")
            try:
                if self.stream and hasattr(self.stream, "stop"):
                    self.stream.stop()
                    self.logger.debug("TextToAudioStream stopped")
                self.is_speaking = False
                return True
            except Exception as e:
                self.logger.error(f"TTS interruption error: {e}")
                if getattr(self.config, "DEBUG", False):
                    self.logger.debug(traceback.format_exc())
        return False

    def shutdown(self):
        """Clean up resources and stop the worker thread."""
        self.logger.info("Shutting down TTS module")
        self.tts_queue.put(None)  # Signal worker thread to stop
        self.worker_thread.join(timeout=5.0)
        if self.worker_thread.is_alive():
            self.logger.warning("Worker thread did not terminate")
        self.logger.debug("TTS module shutdown complete")


class TerminalDisplayModule:
    def __init__(self):
        self.current_sentence = ""
        self.complete_sentences = []
        self.current_style = "cyan bold"
        # ANSI style mapping
        self.style_map = {
            "cyan bold": "\033[36;1m",
            "yellow": "\033[33m",
            "green": "\033[32m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "reset": "\033[0m"
        }

    def start(self):
        """Initialize the display."""
        print("Terminal display started")

    def update_live_text(self, new_text, is_final=False):
        """
        Update the display with new text.
        If 'is_final' is False, update in place.
        If True, print the line and mark it as complete.
        """
        self.current_sentence = new_text
        style_code = self.style_map.get(self.current_style, self.style_map["reset"])
        if is_final:
            # Print final text with a newline and mark sentence complete.
            print(f"{style_code}{new_text}{self.style_map['reset']}")
            self.complete_sentences.append((new_text, self.current_style))
            self.current_sentence = ""
        else:
            # Print updating text on the same line.
            print(f"\r{style_code}{new_text}{self.style_map['reset']}", end='', flush=True)

    def mark_sentence_end(self, sentence, style="green"):
        """
        Mark a sentence as complete with the specified style
        and print it on a new line.
        """
        if sentence.strip():
            self.complete_sentences.append((sentence, style))
            style_code = self.style_map.get(style, self.style_map["reset"])
            print(f"\n{style_code}{sentence}{self.style_map['reset']}")
        self.current_sentence = ""

    def stop(self):
        """Stop the display."""
        print("\nTerminal display stopped")


# Combined Speech Processing System
class SpeechProcessingSystem:
    def __init__(self, vad_name="ISAA0"):
        self.config = Config()
        self.running = False
        self.logger = logging.getLogger("SpeechSystem.Main")

        # Create shared queues
        self.audio_queue = queue.Queue(maxsize=self.config.MAX_QUEUE_SIZE)

        try:
            # Create the display first for callbacks
            self.display = TerminalDisplayModule()

            # Initialize modules with proper dependencies
            self.audio_input = AudioInputModule(self.config, self.audio_queue)
            self.vad = VADModule(self.config)
            self.speech_deserializer = SpeechDeserializerModule(self.config)

            # Initialize TTS with callback references
            self.tts = TTSModule(
                self.config,
                on_tts_start=self._on_tts_start,
                on_tts_end=self._on_tts_end,
                vad_name=vad_name,
            )

            # State tracking
            self.is_tts_active = False
            self.last_speech_time = 0
            self.last_processed_time = 0
            self.current_segment_start_time = 0
            self.segment_timeout = 2.0  # seconds

            # Event loop
            self.loop = asyncio.get_event_loop()

            self.logger.info("Speech processing system initialized")
        except Exception as e:
            self.logger.error(f"Error initializing speech processing system: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())
            raise

    async def start(self, input_mode="microphone", websocket_url=None):
        """Start the speech processing system"""
        try:
            self.running = True
            self.display.start()
            self.logger.info(f"Starting speech processing system with input mode: {input_mode}")

            # Start audio input
            success = False
            if input_mode == "microphone":
                success = await self.audio_input.start_microphone()
            elif input_mode == "websocket":
                if not websocket_url:
                    self.logger.error("Websocket URL is required for websocket input")
                    return False
                success = await self.audio_input.start_websocket(websocket_url)
            else:
                self.logger.error(f"Unsupported input mode: {input_mode}")
                return False

            if not success:
                self.logger.error(f"Failed to start audio input with mode: {input_mode}")
                return False

            # Start processing
            self.logger.debug("Starting audio processing task")
            asyncio.create_task(self._process_audio())

            self.logger.debug("Starting results processing task")
            asyncio.create_task(self._process_results())

            self.logger.info("Speech processing system started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error starting speech processing system: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())
            return False

    async def _process_audio(self):
        """Process audio from the input queue with improved logic"""
        frames_buffer = []  # Buffer to accumulate 20ms frames
        byte_counter = 0
        segment_active = False
        last_active_segment_time = time.time()
        processing_error_count = 0
        current_segment_id = None

        tts_active_recently = False
        tts_cooldown_time = 0.5  # seconds to wait after TTS before processing audio
        last_tts_end_time = 0

        self.logger.debug("Audio processing loop started")

        while self.running:
            try:
                if self.is_tts_active:
                    await asyncio.sleep(0.2)
                    continue
                # Process all available audio in the queue
                audio_processed = False

                while not self.audio_queue.empty():
                    audio_processed = True
                    chunk = self.audio_queue.get()

                    # Generate 20ms frames for VAD
                    frame_size = int(self.config.SAMPLE_RATE * 0.02) * 2  # 20ms of 16-bit samples
                    frames = []

                    # Add to frames buffer
                    frames_buffer.append(chunk)
                    byte_counter += len(chunk)

                    # When we have enough data for at least one 20ms frame
                    while byte_counter >= frame_size:
                        frame_data = b''
                        remaining_bytes = frame_size

                        # Extract a complete frame from the buffer
                        while remaining_bytes > 0 and frames_buffer:
                            buf = frames_buffer[0]
                            if len(buf) <= remaining_bytes:
                                frame_data += buf
                                remaining_bytes -= len(buf)
                                frames_buffer.pop(0)
                            else:
                                frame_data += buf[:remaining_bytes]
                                frames_buffer[0] = buf[remaining_bytes:]
                                remaining_bytes = 0

                        byte_counter -= frame_size
                        frames.append(frame_data)

                    # Process each frame through VAD
                    for frame in frames:
                        if frame is None or len(frame) == 0:
                            continue
                        current_time = time.time()
                        # Check if we're in TTS cooldown period
                        if current_time - last_tts_end_time < tts_cooldown_time:
                            tts_active_recently = True
                            # Clear the audio queue during cooldown to prevent processing TTS echo
                            while not self.audio_queue.empty():
                                self.audio_queue.get()
                            await asyncio.sleep(0.1)
                            continue
                        elif tts_active_recently:
                            # Coming out of cooldown - reset VAD state
                            self.vad.reset()
                            tts_active_recently = False

                        # Skip when TTS is active
                        if self.is_tts_active:
                            last_tts_end_time = current_time  # Update for cooldown
                            await asyncio.sleep(0.1)
                            continue
                        vad_result, segment = self.vad.process_frame(frame)

                        # Logic for handling speech segments
                        if vad_result == "start":
                            # Speech started - call speech start callback
                            self._on_speech_start()
                            segment_active = True
                            self.current_segment_start_time = current_time
                            last_active_segment_time = current_time

                            # Generate a new segment ID for this utterance
                            current_segment_id = f"utterance_{int(current_time)}"

                        elif vad_result == "end" and segment:
                            # Speech ended with a segment to process
                            segment_active = False
                            current_segment_id = None

                            # Process final segment with high priority
                            self.logger.debug(f"Processing final speech segment: {len(segment)} bytes")
                            self.speech_deserializer.process_audio(
                                segment,
                                is_final=True,
                                callback=self._on_transcription_result,
                                segment_id="final"  # Mark as final segment
                            )

                            # Mark end time
                            self.last_speech_time = current_time

                        elif vad_result == "active":
                            # Ongoing speech - update active segment time
                            last_active_segment_time = current_time

                            # Check if we have sliding windows created by VAD
                            if self.vad.sliding_windows and segment_active:
                                # Process the most recent sliding window
                                latest_window = self.vad.sliding_windows[-1]
                                window_id = f"{current_segment_id}_window_{len(self.vad.sliding_windows)}"

                                self.logger.debug(
                                    f"Processing sliding window: {len(latest_window)} bytes, id={window_id}")
                                self.speech_deserializer.process_audio(
                                    latest_window,
                                    is_final=False,  # This is a partial window
                                    callback=self._on_live_update,
                                    segment_id=window_id
                                )

                                # Remove the window from VAD after processing to avoid reprocessing
                                self.vad.sliding_windows.pop()

                            # For live updates, process buffer at intervals (existing code, but add segment_id)
                            time_since_processed = current_time - self.last_processed_time
                            if time_since_processed > 0.5 and len(self.vad.buffer) > self.config.SAMPLE_RATE * 0.5:
                                self.last_processed_time = current_time
                                # Make a copy of the current buffer for live update
                                live_buffer = self.vad.buffer
                                self.logger.debug(f"Processing live update: {len(live_buffer)} bytes")
                                self.speech_deserializer.process_audio(
                                    live_buffer,
                                    is_final=False,
                                    callback=self._on_live_update,
                                    segment_id=f"{current_segment_id}_live_{int(current_time)}" if current_segment_id else None
                                )

                        elif vad_result == "thinking":
                            # User might be thinking - we're in a pause
                            # Update the time but don't end the segment yet
                            pass
                        elif vad_result == "error":
                            # There was an error processing the frame
                            self.logger.warning("Error in VAD frame processing")

                # Check for timeout of active segment
                current_time = time.time()
                if segment_active and (current_time - last_active_segment_time > self.segment_timeout):
                    # Force end segment on timeout
                    segment_active = False
                    if len(self.vad.buffer) > 0:
                        final_segment = self.vad.buffer
                        self.logger.info(
                            f"Speech segment timed out, processing final segment: {len(final_segment)} bytes")
                        self.speech_deserializer.process_audio(
                            final_segment,
                            is_final=True,
                            callback=self._on_transcription_result
                        )
                        self.vad.reset()

                # Reset error count on success
                processing_error_count = 0

                # Avoid CPU spinning - shorter sleep if we processed audio
                await asyncio.sleep(0.01 if audio_processed else 0.05)

            except Exception as e:
                processing_error_count += 1
                self.logger.error(f"Audio processing error: {e}")
                if self.config.DEBUG:
                    self.logger.debug(traceback.format_exc())

                # If too many consecutive errors, reset state
                if processing_error_count > 5:
                    self.logger.warning("Too many processing errors, resetting VAD state")
                    try:
                        self.vad.reset()
                        segment_active = False
                        frames_buffer = []
                        byte_counter = 0
                        processing_error_count = 0
                    except Exception as reset_error:
                        self.logger.error(f"Error resetting VAD: {reset_error}")

                await asyncio.sleep(0.1)

        self.logger.debug("Audio processing loop ended")

    async def _process_results(self):
        """Process transcription results asynchronously"""
        self.logger.debug("Results processing loop started")
        while self.running:
            try:
                await self.speech_deserializer.process_results()
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in results processing: {e}")
                if self.config.DEBUG:
                    self.logger.debug(traceback.format_exc())
                await asyncio.sleep(0.5)
        self.logger.debug("Results processing loop ended")

    def _on_tts_start(self):
        """Callback when TTS starts speaking"""
        self.is_tts_active = True
        self.logger.debug("TTS started speaking")
        # Mute audio input to prevent feedback loop
        self.audio_input.set_mute(True)
        self.vad.reset()

    def _on_tts_end(self, text=None):
        """Callback when TTS finishes speaking"""
        self.is_tts_active = False
        self.logger.debug("TTS finished speaking")
        # Unmute audio input
        # Add a small delay before unmuting to avoid catching the end of TTS

        if text:
            self.speech_deserializer.set_last_tts_output(text)
            # self.display.mark_sentence_end(f"[System]: {text}", style="blue")
        self.loop.call_later(1.0, self._delayed_unmute)

    def _delayed_unmute(self):
        """Delayed unmute to avoid audio feedback"""
        # Unmute audio input after a delay
        self.audio_input.set_mute(False)
        # Reset VAD state after TTS completes
        self.vad.reset()
        # Clear any audio accumulated during TTS
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _on_speech_start(self):
        """Callback for when speech starts"""
        self.logger.info("Speech started")
        try:
            console.print("[green]Speech started...[/green]")
        except Exception as e:
            self.logger.error(f"Error in console output: {e}")

    # In SpeechProcessingSystem class:

    def _on_live_update(self, transcription, is_final):
        """Callback for live transcription updates"""
        if not transcription:
            return

        try:
            # Update VAD with current transcription for intelligent end detection
            self.vad.update_current_transcription(transcription)

            self.display.update_live_text(transcription, is_final)
            if self.config.DEBUG:
                self.logger.debug(f"Live update: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
        except Exception as e:
            self.logger.error(f"Error in live update: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())

    def _on_transcription_result(self, transcription, is_final):
        """Callback for transcription results, distinguishing between live updates and final results"""
        if not transcription:
            return

        try:
            # Update VAD with current transcription for intelligent end detection
            self.vad.update_current_transcription(transcription)
            if is_final:
                self.logger.info(f"Final transcription: {transcription}")

                # Mark the sentence as complete in the display
                self.display.mark_sentence_end(f"[User]: {transcription}", style="green")
                print(f"[User]: {transcription}")
                # Make sure we're using the proper concatenated text, not just the last chunk
                response = f"USER: {transcription}"

                # Log the final response text being sent to TTS
                self.logger.debug(f"Sending to TTS: {response}")
                # self.tts._transmutate(response)
                self.tts.speak(
                    response,
                    callback_start=lambda: self.logger.debug("TTS response started"),
                    callback_end=lambda text: self._on_tts_end() and self.logger.debug(
                        f"TTS response finished: {text[:50]}{'...' if len(text) > 50 else ''}")
                )

                # Reset VAD transcription after final response
                self.vad.update_current_transcription("")
            else:
                # Live update
                if self.config.DEBUG:
                    self.logger.debug(
                        f"Live transcription update: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
                self.display.update_live_text(transcription, is_final=False)
        except Exception as e:
            self.logger.error(f"Error in transcription callback: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())

    async def stop(self):
        """Stop the speech processing system and clean up resources"""
        self.logger.info("Stopping speech processing system")
        self.running = False

        try:
            await self.audio_input.stop()
            self.speech_deserializer.shutdown()
            self.tts.shutdown()
            self.display.stop()
            self.logger.info("Speech processing system stopped")
        except Exception as e:
            self.logger.error(f"Error shutting down system: {e}")
            if self.config.DEBUG:
                self.logger.debug(traceback.format_exc())

# Main entry point
async def main():
    # Process command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Speech Processing System')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug logging')
    parser.add_argument('-m')
    parser.add_argument('-n', '--name',metavar="name",
                        type=str,
                        help="Specify an id for the ToolBox instance",
                        default="main")
    parser.add_argument('--input', choices=['microphone', 'websocket'], default='microphone',
                        help='Audio input mode (microphone or websocket)')
    parser.add_argument('--websocket-url', help='WebSocket URL for audio input')
    args = parser.parse_args()

    # Set debug mode based on arguments or environment
    debug_mode = args.debug or (os.getenv("DEBUG", "0") == "1")
    Config.set_debug(debug_mode)
    if not debug_mode:
        logging.disable(logging.CRITICAL)
    await get_app().get_mod("isaa")
    get_app().get_mod("isaa").init_isaa(build=True)

    # Create and start the speech processing system
    try:
        system = SpeechProcessingSystem("ISAA0" if args.name == "main" else args.name)

        console.print("[bold yellow]Starting speech processing system...[/bold yellow]")
        logger.info(f"Starting system with input mode: {args.input}, debug: {debug_mode}")

        # Set input parameters
        input_mode = args.input
        websocket_url = args.websocket_url

        # Validate websocket URL if needed
        if input_mode == "websocket" and not websocket_url:
            logger.error("Websocket URL is required for websocket input mode")
            console.print("[bold red]Error: Websocket URL is required for websocket input mode[/bold red]")
            return

        # Start the system
        if await system.start(input_mode, websocket_url):
            console.print("[bold green]System started successfully![/bold green]")
            console.print("[bold white]Speak into the microphone. Press Ctrl+C to exit.[/bold white]")

            try:
                # Keep the main task running until interrupted
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping system")
                console.print("[bold yellow]Stopping system...[/bold yellow]")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if debug_mode:
                    logger.debug(traceback.format_exc())
            finally:
                await system.stop()
        else:
            logger.error("Failed to start speech processing system")
            console.print("[bold red]Failed to start speech processing system[/bold red]")

        console.print("[bold green]Demo Completed Successfully![/bold green]")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if debug_mode:
            logger.debug(traceback.format_exc())
        console.print(f"[bold red]Fatal error: {e}[/bold red]")


async def run(_, __):
    global console

    # Initialize the rich console for colored output
    from rich.console import Console
    console = Console()
    # sys.argv += ['--debug']
    await main()


NAME = "VAD"

if __name__ == "__main__":

    # Run the main function using asyncio
    try:
        #
        #
        # sys.argv += ['--debug']
        sys.argv += ['-n','demo3_vad']
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal exception in main: {e}")
        logger.debug(traceback.format_exc())
        print(f"Fatal error: {e}")
        sys.exit(1)

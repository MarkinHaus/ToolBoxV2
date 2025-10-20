# toolboxv2.utils.extras.notification
import sys
import subprocess
import os
import threading
import time
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
import queue

from toolboxv2.utils.singelton_class import Singleton
from toolboxv2.utils.extras.Style import text_save
from toolboxv2.utils.security.cryp import Code
chastext = {}


@dataclass
class NotificationAction:
    """Represents an action button in a notification"""
    id: str
    label: str
    callback: Optional[Callable[[], Any]] = None
    is_default: bool = False


@dataclass
class NotificationDetails:
    """Expandable details for notifications"""
    title: str
    content: str
    data: Optional[Dict] = None


class NotificationType(Enum):
    """Types of notifications"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    QUESTION = "question"

class NotificationPosition(Enum):
    """Position options for notifications"""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class NotificationSystem(metaclass=Singleton):
    """
    Cross-platform notification system with OS integration and tkinter fallback
    """

    def __init__(self):
        self.platform = sys.platform.lower()
        self.fallback_to_tkinter = True
        self.sound_enabled = False
        self.default_timeout = 1500 # Add default timeout in milliseconds
        self.max_timeout = 30000
        self.default_position = NotificationPosition.TOP_RIGHT
        self._test_os_notifications()

    def _test_os_notifications(self):
        """Test if OS notifications are available"""
        try:
            if self.platform.startswith('win'):
                # Test Windows toast notifications
                try:
                    import win10toast
                    # Test if we can create a ToastNotifier without errors
                    try:
                        toaster = win10toast.ToastNotifier()
                        # Test if classAtom exists (common error)
                        if not hasattr(toaster, 'classAtom'):
                            print("⚠️  win10toast library has compatibility issues. Will use alternative methods.")
                            # Don't set fallback_to_tkinter = True, we have alternatives
                        self.fallback_to_tkinter = False
                    except AttributeError:
                        print("⚠️  win10toast library has issues. Using alternative Windows notification methods.")
                        self.fallback_to_tkinter = True
                except ImportError:
                    print("⚠️  Windows toast notifications not available. Install win10toast: pip install win10toast")
                    print("    Alternative: Will try built-in Windows notification methods.")
                    self.fallback_to_tkinter = True  # We still have alternatives
            elif self.platform.startswith('darwin'):
                # Test macOS notifications
                try:
                    result = subprocess.run(['which', 'osascript'],
                                            capture_output=True, text=True)
                    if result.returncode != 0:
                        raise FileNotFoundError
                    self.fallback_to_tkinter = False
                except:
                    print("⚠️  macOS notifications not available. osascript not found.")
                    self.fallback_to_tkinter = True

            elif self.platform.startswith('linux'):
                # Test Linux notifications
                try:
                    result = subprocess.run(['which', 'notify-send'],
                                            capture_output=True, text=True)
                    if result.returncode != 0:
                        raise FileNotFoundError
                    self.fallback_to_tkinter = False
                except:
                    print(
                        "⚠️  Linux notifications not available. Install libnotify-bin: sudo apt install libnotify-bin")
                    self.fallback_to_tkinter = True
            else:
                print("⚠️  Unknown platform. Using tkinter fallback.")
                self.fallback_to_tkinter = True

        except Exception as e:
            print(f"⚠️  OS notification test failed: {e}. Using tkinter fallback.")
            self.fallback_to_tkinter = True

    def show_notification(self,
                          title: str,
                          message: str,
                          notification_type: NotificationType = NotificationType.INFO,
                          actions: List[NotificationAction] = None,
                          details: NotificationDetails = None,
                          timeout: int = None,
                          play_sound: bool = False,
                          position: NotificationPosition = None) -> Optional[str]:
        """
        Show a notification with optional actions and details

        Args:
            title (str): Title of the notification
            message (str): Main message of the notification
            notification_type (NotificationType): Type of notification
            actions (List[NotificationAction]): List of action buttons
            details (NotificationDetails): Expandable details
            timeout (int): Timeout in milliseconds
            play_sound (bool): Whether to play a sound
            position (NotificationPosition): Position on screen

        Returns the ID of the selected action, or None if dismissed
        """
        # Handle position configuration
        if position is None:
            position = self.default_position

        if timeout is None:
            timeout = self.default_timeout
        elif timeout > self.max_timeout:
            timeout = self.max_timeout
        elif timeout < 0:
            timeout = 0

        if play_sound and self.sound_enabled:
            self._play_notification_sound(notification_type)

        if self.fallback_to_tkinter or actions or details:
            # Use tkinter for complex notifications or as fallback
            return self._show_tkinter_notification(title, message, notification_type,
                                                   actions, details, timeout, position)
        else:
            # Use OS notification for simple notifications
            return self._show_os_notification(title, message, notification_type, timeout)

    def set_default_timeout(self, timeout_ms: int):
        """Set default timeout for notifications"""
        if timeout_ms < 0:
            self.default_timeout = 0  # No timeout
        elif timeout_ms > self.max_timeout:
            self.default_timeout = self.max_timeout
        else:
            self.default_timeout = timeout_ms

    def set_max_timeout(self, max_timeout_ms: int):
        """Set maximum allowed timeout"""
        if max_timeout_ms > 0:
            self.max_timeout = max_timeout_ms

    def set_default_position(self, position: NotificationPosition):
        """Set default position for notifications"""
        self.default_position = position

    def _show_os_notification(self, title: str, message: str,
                              notification_type: NotificationType, timeout: int) -> None:
        """Show OS native notification"""

        try:
            if self.platform.startswith('win'):
                self._show_windows_notification(title, message, notification_type, timeout)
            elif self.platform.startswith('darwin'):
                self._show_macos_notification(title, message, notification_type, timeout)
            elif self.platform.startswith('linux'):
                self._show_linux_notification(title, message, notification_type, timeout)
        except Exception as e:
            print(f"⚠️  OS notification failed: {e}. Falling back to tkinter.")
            return self._show_tkinter_notification(title, message, notification_type)

    def _show_windows_notification(self, title: str, message: str,
                                               notification_type: NotificationType, timeout):
        """Alternative Windows notification using ctypes"""
        try:
            import ctypes
            from ctypes import wintypes

            # Try using Windows 10+ notification API via PowerShell
            try:
                icon_map = {
                    NotificationType.INFO: "Information",
                    NotificationType.SUCCESS: "success",
                    NotificationType.WARNING: "Warning",
                    NotificationType.ERROR: "Error",
                    NotificationType.QUESTION: "Question"
                }

                icon_type = icon_map.get(notification_type, "Information")

                # PowerShell script to show notification
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                [Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

                $template = @"
                <toast>
                    <visual>
                        <binding template="ToastGeneric">
                            <text>{title}</text>
                            <text>{message}</text>
                        </binding>
                    </visual>
                </toast>
                "@

                $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
                $xml.LoadXml($template)
                $toast = New-Object Windows.UI.Notifications.ToastNotification $xml
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Python App").Show($toast)
                '''

                result = subprocess.run(['powershell', '-Command', text_save(ps_script)],
                                        capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    return

            except Exception:
                pass

            # Fallback to simple MessageBox
            MB_ICONINFORMATION = 0x40
            MB_ICONWARNING = 0x30
            MB_ICONERROR = 0x10
            MB_ICONQUESTION = 0x20

            icon_map = {
                NotificationType.INFO: MB_ICONINFORMATION,
                NotificationType.SUCCESS: MB_ICONINFORMATION,
                NotificationType.WARNING: MB_ICONWARNING,
                NotificationType.ERROR: MB_ICONERROR,
                NotificationType.QUESTION: MB_ICONQUESTION
            }

            icon = icon_map.get(notification_type, MB_ICONINFORMATION)

            # Show MessageBox in separate thread to avoid blocking
            def show_messagebox():
                try:
                    ctypes.windll.user32.MessageBoxW(0, message, title, icon)
                except:
                    pass

            threading.Thread(target=show_messagebox, daemon=True).start()

        except Exception:
            raise Exception("All Windows notification methods failed")

    def _show_macos_notification(self, title: str, message: str,
                                 notification_type: NotificationType, timeout: int):
        """macOS notification using osascript"""
        try:
            script = f'''
                display notification "{message}" with title "{title}"
            '''
            subprocess.run(['osascript', '-e', text_save(script)], check=True)
        except Exception as e:
            raise Exception(f"macOS notification failed: {e}")

    def _show_linux_notification(self, title: str, message: str,
                                 notification_type: NotificationType, timeout: int):
        """Linux notification using notify-send"""
        try:
            urgency = "normal"
            if notification_type == NotificationType.ERROR:
                urgency = "critical"
            elif notification_type == NotificationType.WARNING:
                urgency = "normal"

            icon = self._get_linux_icon(notification_type)

            subprocess.run([
                'notify-send',
                f'--urgency={urgency}',
                f'--expire-time={timeout}',
                f'--icon={icon}',
                text_save(title),
                text_save(message)
            ], check=True)
        except Exception as e:
            raise Exception(f"Linux notification failed: {e}")

    def _show_tkinter_notification(self, title: str, message: str,
                                   notification_type: NotificationType,
                                   actions: List[NotificationAction] = None,
                                   details: NotificationDetails = None,
                                   timeout: int = 5000,
                                   position: NotificationPosition = NotificationPosition.CENTER) -> Optional[str]:
        """Modern dark-themed tkinter notification dialog"""

        # Use a queue to communicate between threads
        result_queue = queue.Queue()

        def run_notification():
            try:
                import tkinter as tk
                from tkinter import ttk

                # Create root window
                root = tk.Tk()
                root.withdraw()  # Hide the root window

                # Create notification window
                window = tk.Toplevel(root)

                # Dark theme colors
                bg_color = "#2b2b2b"
                fg_color = "#ffffff"
                accent_color = self._get_accent_color(notification_type)
                button_color = "#404040"
                button_hover = "#505050"
                border_color = "#404040"

                # Remove window decorations for custom styling
                window.overrideredirect(True)
                window.configure(bg=border_color)

                # Variables for dragging (use instance variables to avoid threading issues)
                window.drag_data = {"x": 0, "y": 0}
                window.details_visible = False
                window.result = None

                # Create main container with border
                border_frame = tk.Frame(window, bg=border_color, padx=1, pady=1)
                border_frame.pack(fill=tk.BOTH, expand=True)

                main_container = tk.Frame(border_frame, bg=bg_color)
                main_container.pack(fill=tk.BOTH, expand=True)

                # Title bar for dragging and close button
                title_bar = tk.Frame(main_container, bg=accent_color, height=25)
                title_bar.pack(fill=tk.X, side=tk.TOP)
                title_bar.pack_propagate(False)

                # Window title in title bar
                title_label = tk.Label(title_bar, text="Notification",
                                       font=("Arial", 9), bg=accent_color, fg=fg_color)
                title_label.pack(side=tk.LEFT, padx=8, pady=4)

                # Close button
                def close_window():
                    window.result = None
                    result_queue.put(window.result)
                    root.quit()
                    root.destroy()

                close_btn = tk.Label(title_bar, text="✕", font=("Arial", 10, "bold"),
                                     bg=accent_color, fg=fg_color, cursor="hand2",
                                     padx=8, pady=2)
                close_btn.pack(side=tk.RIGHT)
                close_btn.bind("<Button-1>", lambda e: close_window())
                close_btn.bind("<Enter>", lambda e: close_btn.config(bg=self._lighten_color(accent_color, -0.2)))
                close_btn.bind("<Leave>", lambda e: close_btn.config(bg=accent_color))

                # Make title bar draggable
                def start_drag(event):
                    window.drag_data["x"] = event.x
                    window.drag_data["y"] = event.y

                def on_drag(event):
                    x = window.winfo_x() + (event.x - window.drag_data["x"])
                    y = window.winfo_y() + (event.y - window.drag_data["y"])
                    window.geometry(f"+{x}+{y}")

                title_bar.bind("<Button-1>", start_drag)
                title_bar.bind("<B1-Motion>", on_drag)
                title_label.bind("<Button-1>", start_drag)
                title_label.bind("<B1-Motion>", on_drag)

                # Content frame
                content_frame = tk.Frame(main_container, bg=bg_color, padx=15, pady=12)
                content_frame.pack(fill=tk.BOTH, expand=True)

                # Header with icon and title (more compact)
                header_frame = tk.Frame(content_frame, bg=bg_color)
                header_frame.pack(fill=tk.X, pady=(0, 8))

                # Notification type icon (smaller)
                icon_label = tk.Label(header_frame, text=self._get_emoji_icon(notification_type),
                                      font=("Arial", 16), bg=bg_color, fg=accent_color)
                icon_label.pack(side=tk.LEFT, padx=(0, 8))

                # Title (smaller font)
                title_text = tk.Label(header_frame, text=title, font=("Arial", 11, "bold"),
                                      bg=bg_color, fg=fg_color, wraplength=280)
                title_text.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")

                # Message (more compact)
                message_label = tk.Label(content_frame, text=message, font=("Arial", 9),
                                         bg=bg_color, fg=fg_color, wraplength=320, justify=tk.LEFT)
                message_label.pack(fill=tk.X, pady=(0, 8))

                # Details section (expandable) - initially hidden
                details_frame = None
                details_text_widget = None

                if details:
                    details_container = tk.Frame(content_frame, bg=bg_color)
                    details_container.pack(fill=tk.X, pady=(0, 8))

                    def toggle_details():
                        nonlocal details_frame, details_text_widget

                        if not window.details_visible:
                            # Show details
                            if details_frame is None:
                                details_frame = tk.Frame(details_container, bg=bg_color)
                                details_frame.pack(fill=tk.X, pady=(4, 0))

                                # Create scrollable text area
                                text_frame = tk.Frame(details_frame, bg="#1e1e1e")
                                text_frame.pack(fill=tk.X, pady=(0, 0))

                                details_text_widget = tk.Text(text_frame, height=5, bg="#1e1e1e", fg=fg_color,
                                                              border=0, wrap=tk.WORD, font=("Consolas", 8),
                                                              padx=8, pady=6)
                                details_text_widget.pack(fill=tk.X)

                                detail_content = f"{details.title}\n{'-' * min(40, len(details.title))}\n{details.content}"
                                if details.data:
                                    detail_content += f"\n\nData:\n{json.dumps(details.data, indent=2)}"

                                details_text_widget.insert(tk.END, detail_content)
                                details_text_widget.config(state=tk.DISABLED)

                            details_btn.config(text="▼ Hide Details")
                            details_frame.pack(fill=tk.X, pady=(4, 0))
                            window.details_visible = True

                            # Resize window
                            window.update_idletasks()
                            new_height = window.winfo_reqheight()
                            window.geometry(f"380x{new_height}")
                        else:
                            # Hide details
                            details_btn.config(text="▶ Show Details")
                            if details_frame:
                                details_frame.pack_forget()
                            window.details_visible = False

                            # Resize window back
                            window.update_idletasks()
                            new_height = window.winfo_reqheight()
                            window.geometry(f"380x{new_height}")

                    details_btn = tk.Button(details_container, text="▶ Show Details",
                                            command=toggle_details, bg=button_color, fg=fg_color,
                                            border=0, font=("Arial", 8), relief=tk.FLAT, cursor="hand2")
                    details_btn.pack(anchor=tk.W)

                # Action buttons (more compact)
                if actions:
                    button_frame = tk.Frame(content_frame, bg=bg_color)
                    button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))

                    for i, action in enumerate(actions):
                        def make_callback(action_id, callback):
                            def callback_wrapper():
                                window.result = action_id
                                result_queue.put(window.result)
                                if callback:
                                    # Run callback in separate thread
                                    threading.Thread(target=callback, daemon=True).start()
                                root.quit()
                                root.destroy()

                            return callback_wrapper

                        btn_bg = accent_color if action.is_default else button_color
                        btn = tk.Button(button_frame, text=action.label,
                                        command=make_callback(action.id, action.callback),
                                        bg=btn_bg, fg=fg_color, border=0,
                                        font=("Arial", 9), relief=tk.FLAT,
                                        padx=12, pady=6, cursor="hand2")
                        btn.pack(side=tk.RIGHT, padx=(4, 0))

                        # Hover effects
                        def on_enter(e, btn=btn, color=btn_bg):
                            btn.config(bg=self._lighten_color(color))

                        def on_leave(e, btn=btn, color=btn_bg):
                            btn.config(bg=color)

                        btn.bind("<Enter>", on_enter)
                        btn.bind("<Leave>", on_leave)
                else:
                    # Default OK button
                    button_frame = tk.Frame(content_frame, bg=bg_color)
                    button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(8, 0))

                    def ok_clicked():
                        window.result = "ok"
                        result_queue.put(window.result)
                        root.quit()
                        root.destroy()

                    ok_btn = tk.Button(button_frame, text="OK", command=ok_clicked,
                                       bg=accent_color, fg=fg_color, border=0,
                                       font=("Arial", 9), relief=tk.FLAT,
                                       padx=12, pady=6, cursor="hand2")
                    ok_btn.pack(side=tk.RIGHT)

                # Set initial window size (slimmer)
                base_height = 150
                if timeout > 5000:
                    base_height += 20  # Add space for timeout indicator

                # Center window on screen
                # Position window based on specified position
                window.update()
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                window_width = 380
                window_height = window.winfo_height()

                # Calculate position based on enum
                margin = 20  # Margin from screen edges

                if position == NotificationPosition.TOP_LEFT:
                    x, y = margin, margin
                elif position == NotificationPosition.TOP_CENTER:
                    x, y = (screen_width - window_width) // 2, margin
                elif position == NotificationPosition.TOP_RIGHT:
                    x, y = screen_width - window_width - margin, margin
                elif position == NotificationPosition.CENTER_LEFT:
                    x, y = margin, (screen_height - window_height) // 2
                elif position == NotificationPosition.CENTER:
                    x, y = (screen_width - window_width) // 2, (screen_height - window_height) // 2 - 50
                elif position == NotificationPosition.CENTER_RIGHT:
                    x, y = screen_width - window_width - margin, (screen_height - window_height) // 2
                elif position == NotificationPosition.BOTTOM_LEFT:
                    x, y = margin, screen_height - window_height - margin - 50  # Account for taskbar
                elif position == NotificationPosition.BOTTOM_CENTER:
                    x, y = (screen_width - window_width) // 2, screen_height - window_height - margin - 50
                elif position == NotificationPosition.BOTTOM_RIGHT:
                    x, y = screen_width - window_width - margin, screen_height - window_height - margin - 50
                else:
                    # Default to center
                    x, y = (screen_width - window_width) // 2, (screen_height - window_height) // 2 - 50
                window.geometry(f"{window_width}x{window_height}+{x}+{y}")
                window.update()
                # Always on top and focus
                window.attributes('-topmost', True)
                window.focus_force()

                # Auto-close after timeout (if no actions)
                if not actions:
                    # Auto-close after timeout (for all notifications if timeout > 0)
                    if timeout > 0:
                        def create_auto_close():
                            def auto_close_handler():
                                try:
                                    if root.winfo_exists():
                                        window.result = 'timeout'
                                        result_queue.put('timeout')
                                        root.quit()
                                        root.destroy()
                                except tk.TclError:
                                    pass  # Window already destroyed
                                except Exception:
                                    pass  # Handle any other errors silently

                            root.after(timeout, auto_close_handler)

                        create_auto_close()

                    # Add timeout indicator if timeout > 10 seconds
                    # Alternative: Progress bar timeout indicator (replace the text version above)
                    if timeout > 5000:
                        timeout_frame = tk.Frame(content_frame, bg=bg_color)
                        timeout_frame.pack(fill=tk.X, pady=(2, 4))

                        # Progress bar for visual timeout
                        progress_bg = tk.Frame(timeout_frame, bg="#444444", height=4)
                        progress_bg.pack(fill=tk.X, pady=(0, 2))

                        progress_bar = tk.Frame(progress_bg, bg="#666666", height=4)
                        progress_bar.place(x=0, y=0, relwidth=1.0, height=4)

                        # Timeout text
                        timeout_label = tk.Label(timeout_frame,
                                                 text=f"⏱️ Auto-closes in {timeout // 1000}s",
                                                 font=("Arial", 8), bg=bg_color, fg="#888888")
                        timeout_label.pack(anchor=tk.E)

                        def setup_progress_countdown():
                            total_time = timeout // 1000
                            remaining = [total_time]

                            def update_progress():
                                try:
                                    if remaining[0] > 0 and root and root.winfo_exists():
                                        # Update text
                                        timeout_label.config(text=f"⏱️ Auto-closes in {remaining[0]}s")

                                        # Update progress bar
                                        progress_width = remaining[0] / total_time
                                        progress_bar.place(relwidth=progress_width)

                                        remaining[0] -= 1
                                        root.after(1000, update_progress)
                                    elif root and root.winfo_exists():
                                        timeout_label.config(text="⏱️ Closing...")
                                        progress_bar.place(relwidth=0)
                                except (tk.TclError, AttributeError):
                                    pass

                            root.after(1000, update_progress)

                        setup_progress_countdown()

                # Handle escape key
                def on_escape(event):
                    close_window()

                window.bind('<Escape>', on_escape)
                window.focus_set()

                # Start the GUI main loop
                root.mainloop()

            except Exception as e:
                print(f"⚠️  Tkinter notification error: {e}")
                result_queue.put(None)

        # Run notification in the main thread if possible, otherwise in a separate thread
        if threading.current_thread() is threading.main_thread():
            run_notification()
        else:
            # If not in main thread, we need to handle this differently
            gui_thread = threading.Thread(target=run_notification, daemon=True)
            gui_thread.start()
            gui_thread.join(timeout=30)  # Don't wait forever

        # Get result from queue
        try:
            if actions:
                return result_queue.get(timeout=1)
            return None
        except queue.Empty:
            return None

    def _play_notification_sound(self, notification_type: NotificationType):
        """Play appropriate sound for notification type"""
        try:
            if notification_type == NotificationType.ERROR:
                self._play_sound(frequency=800, duration=0.5)
            elif notification_type == NotificationType.WARNING:
                self._play_sound(frequency=600, duration=0.3)
            elif notification_type == NotificationType.SUCCESS:
                self._play_sound(frequency=1000, duration=0.2)
            else:
                self._play_sound(frequency=700, duration=0.3)
        except:
            pass  # Don't let sound errors break notifications

    def _play_sound(self, frequency: int = 800, duration: float = 0.3):
        """Play notification sound"""

        def play():
            try:
                if self.platform.startswith('win'):
                    import winsound
                    winsound.Beep(frequency, int(duration * 1000))
                elif self.platform.startswith('darwin'):
                    subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'],
                                   check=True, capture_output=True)
                elif self.platform.startswith('linux'):
                    try:
                        subprocess.run(['paplay', '--raw', '--format=s16le',
                                        '--rate=44100', '--channels=1'],
                                       input=self._generate_tone_data(frequency, duration, 44100),
                                       check=True, timeout=2)
                    except:
                        print('\a')  # Fallback to system bell
                else:
                    print('\a')
            except:
                print('\a')  # Ultimate fallback

        # Play sound in separate thread to not block UI
        threading.Thread(target=play, daemon=True).start()

    def _generate_tone_data(self, frequency: int, duration: float, sample_rate: int = 44100) -> bytes:
        """Generate raw audio data for a sine wave tone"""
        import math
        import struct

        num_samples = int(sample_rate * duration)
        tone_data = []

        for i in range(num_samples):
            t = i / sample_rate
            fade = min(1.0, t * 10, (duration - t) * 10)
            sample = int(16384 * fade * math.sin(2 * math.pi * frequency * t))
            tone_data.append(struct.pack('<h', sample))

        return b''.join(tone_data)

    def _get_emoji_icon(self, notification_type: NotificationType) -> str:
        """Get emoji icon for notification type"""
        icons = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.QUESTION: "❓"
        }
        return icons.get(notification_type, "📢")

    def _get_accent_color(self, notification_type: NotificationType) -> str:
        """Get accent color for notification type"""
        colors = {
            NotificationType.INFO: "#3498db",
            NotificationType.SUCCESS: "#27ae60",
            NotificationType.WARNING: "#f39c12",
            NotificationType.ERROR: "#e74c3c",
            NotificationType.QUESTION: "#9b59b6"
        }
        return colors.get(notification_type, "#3498db")

    def _lighten_color(self, color: str, factor: float = 0.2) -> str:
        """Lighten or darken a hex color"""
        try:
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
            if factor > 0:
                # Lighten
                rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
            else:
                # Darken
                rgb = tuple(max(0, int(c * (1 + factor))) for c in rgb)
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        except:
            return color

    def _get_icon_path(self, notification_type: NotificationType) -> Optional[str]:
        """Get icon path for Windows notifications"""
        return None

    def _get_linux_icon(self, notification_type: NotificationType) -> str:
        """Get Linux system icon name"""
        icons = {
            NotificationType.INFO: "dialog-information",
            NotificationType.SUCCESS: "dialog-information",
            NotificationType.WARNING: "dialog-warning",
            NotificationType.ERROR: "dialog-error",
            NotificationType.QUESTION: "dialog-question"
        }
        return icons.get(notification_type, "dialog-information")


# Convenience functions
def create_notification_system() -> NotificationSystem:
    """Create and return a notification system instance"""
    return NotificationSystem()


def quick_info(title: str, message: str, **kwargs):
    """Quick info notification"""
    notifier = create_notification_system()
    notifier.show_notification(title, message, NotificationType.INFO, **kwargs)


def quick_success(title: str, message: str, **kwargs):
    """Quick success notification"""
    notifier = create_notification_system()
    notifier.show_notification(title, message, NotificationType.SUCCESS, **kwargs)


def quick_warning(title: str, message: str, **kwargs):
    """Quick warning notification"""
    notifier = create_notification_system()
    notifier.show_notification(title, message, NotificationType.WARNING, **kwargs)


def quick_error(title: str, message: str, **kwargs):
    """Quick error notification"""
    notifier = create_notification_system()
    notifier.show_notification(title, message, NotificationType.ERROR, **kwargs)


def ask_question(title: str, message: str,
                 yes_callback: Callable = None,
                 no_callback: Callable = None, **kwargs) -> Optional[str]:
    """Ask a yes/no question"""
    notifier = create_notification_system()

    actions = [
        NotificationAction("yes", "Yes", yes_callback, is_default=True),
        NotificationAction("no", "No", no_callback)
    ]

    return notifier.show_notification(
        title, message, NotificationType.QUESTION, actions=actions, **kwargs
    )


# Example usage
def example_notifications():
    """Example notification scenarios with better timing"""

    notifier = create_notification_system()

    # Simple notification
    print("1. Simple info notification...")
    notifier.show_notification(
        title="Welcome!",
        message="Application started successfully.",
        notification_type=NotificationType.INFO
    )

    time.sleep(2)

    # Success notification
    print("2. Success notification...")
    notifier.show_notification(
        title="Task Complete",
        message="Your file has been processed successfully.",
        notification_type=NotificationType.SUCCESS
    )

    time.sleep(2)

    # Warning with details
    print("3. Warning with expandable details...")
    details = NotificationDetails(
        title="Performance Warning",
        content="The system is running low on memory. Consider closing some applications to free up resources.",
        data={
            "memory_usage": "85%",
            "available_memory": "2.1 GB",
            "total_memory": "16 GB",
            "top_processes": ["Chrome", "Visual Studio", "Photoshop"]
        }
    )

    notifier.show_notification(
        title="System Warning",
        message="High memory usage detected.",
        notification_type=NotificationType.WARNING,
        details=details
    )

    time.sleep(2)

    # Interactive notification with actions
    print("4. Interactive notification with actions...")

    def handle_update():
        print("🔄 Update initiated!")
        time.sleep(1)
        notifier.show_notification(
            title="Update Complete",
            message="Application has been updated to version 2.1.0.",
            notification_type=NotificationType.SUCCESS
        )

    def handle_remind_later():
        print("⏰ Reminder set for later!")
        notifier.show_notification(
            title="Reminder Set",
            message="You'll be reminded about the update in 1 hour.",
            notification_type=NotificationType.INFO
        )

    actions = [
        NotificationAction("update", "Update Now", handle_update, is_default=True),
        NotificationAction("later", "Remind Later", handle_remind_later),
        NotificationAction("skip", "Skip Version", lambda: print("❌ Update skipped"))
    ]

    selected_action = notifier.show_notification(
        title="Update Available",
        message="Version 2.1.0 is ready to install with bug fixes and new features.",
        notification_type=NotificationType.QUESTION,
        actions=actions,
        details=NotificationDetails(
            title="Update Information",
            content="This update includes security patches, performance improvements, and new features.",
            data={
                "version": "2.1.0",
                "size": "25.3 MB",
                "release_date": "2024-01-15",
                "changelog": [
                    "Fixed memory leak in file processing",
                    "Added dark mode support",
                    "Improved startup time by 40%",
                    "Updated dependencies for security"
                ]
            }
        )
    )

    print(f"✅ Selected action: {selected_action}")

    print("5. Testing different positions...")

    positions_to_test = [
        (NotificationPosition.TOP_RIGHT, "Top Right"),
        (NotificationPosition.BOTTOM_LEFT, "Bottom Left"),
        (NotificationPosition.TOP_CENTER, "Top Center"),
        (NotificationPosition.CENTER_RIGHT, "Center Right")
    ]
    notifier.fallback_to_tkinter = True
    for position, pos_name in positions_to_test:
        notifier.show_notification(
            title=f"{pos_name} Notification",
            message=f"This notification appears at {pos_name.lower()}",
            notification_type=NotificationType.INFO,
            position=position,
            timeout=2000
        )
        time.sleep(0.5)


if __name__ == "__main__":
    print("🔔 Modern Notification System Demo")
    print("=" * 50)

    # Test quick functions
    print("\n📱 Testing quick notification functions...")
    quick_info("Quick Test", "This is a quick info message!")
    time.sleep(1)

    quick_success("Success!", "Operation completed successfully!")
    time.sleep(1)

    # Test question
    print("\n❓ Testing question dialog...")
    result = ask_question(
        "Confirmation",
        "Do you want to continue with this action?",
        yes_callback=lambda: print("✅ User confirmed!"),
        no_callback=lambda: print("❌ User cancelled!")
    )
    print(f"Question result: {result}")

    # Run full examples
    print("\n🎯 Running full notification examples...")
    example_notifications()

    print("\n✨ Demo complete!")

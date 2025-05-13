
emoji_set_work_phases = ["📂", "📄", "🖊️", "📊", "📈", "💻", "🔧", "✅", "🚀", "🏁"]
emoji_set_thermometer = ["❄️", "🌬️", "☁️", "🌤️", "🌞", "🔥", "🌋", "💥", "🌟", "☀️"]

import logging
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ProgressMessenger:
    def __init__(self, messenger, recipient_phone: str, max_steps: int = 5, emoji_set: list[str] = None, content=None):
        self.messenger = messenger
        self.recipient_phone = recipient_phone
        self.max_steps = max_steps
        self.emoji_set = emoji_set or ["⬜", "⬛", "🟩", "🟨", "🟦"]
        self.message_id = None
        self.content = content

    def send_initial_message(self, mode: str = "progress"):
        """
        Sends the initial message. Modes can be 'progress' or 'loading'.
        """
        if mode == "progress":
            emoji_legend = "\n".join(
                f"{emoji} - Step {i + 1}" for i, emoji in enumerate(self.emoji_set)
            )
            content = (
                "Progress is being updated in real-time!\n\n"
                "Legend:\n"
                f"{emoji_legend}\n\n"
                "Stay tuned for updates!"
            )
        elif mode == "loading":
            content = (
                "Loading in progress! 🌀\n"
                "The indicator will loop until work is done."
            )
        else:
            raise ValueError("Invalid mode. Use 'progress' or 'loading'.")

        if self.content is not None:
            content += '\n'+self.content
        message = self.messenger.create_message(content=content, to=self.recipient_phone)
        response = message.send(sender=0)
        self.message_id = response.get("messages", [{}])[0].get("id")
        logging.info(f"Initial message sent: {content}")
        return self.message_id

    def update_progress(self, step_flag: threading.Event):
        """
        Updates the reaction on the message to represent progress.
        """
        if not self.message_id:
            raise ValueError("Message ID not found. Ensure the initial message is sent first.")
        message = self.messenger.create_message(id=self.message_id, to=self.recipient_phone)
        for step in range(self.max_steps):
            emoji = self.emoji_set[step % len(self.emoji_set)]
            message.react(emoji)
            logging.info(f"Progress updated: Step {step + 1}/{self.max_steps} with emoji {emoji}")
            while not step_flag.is_set():
                time.sleep(0.5)
            step_flag.clear()
        # Final acknowledgment
        message.react("👍")
        logging.info("Progress completed with final acknowledgment.")

    def update_loading(self, stop_flag: threading.Event):
        """
        Continuously updates the reaction to represent a looping 'loading' indicator.
        """
        if not self.message_id:
            raise ValueError("Message ID not found. Ensure the initial message is sent first.")
        message = self.messenger.create_message(id=self.message_id, to=self.recipient_phone)
        step = 0
        while not stop_flag.is_set():
            emoji = self.emoji_set[step % len(self.emoji_set)]
            message.react(emoji)
            logging.info(f"Loading update: {emoji}")
            time.sleep(1)  # Faster updates for loading
            step += 1
        # Final acknowledgment
        message.react("✅")
        logging.info("Loading completed with final acknowledgment.")
        message.reply("✅Done✅")

    def start_progress_in_background(self, step_flag):
        """
        Starts the progress update in a separate thread.
        """
        threading.Thread(target=self.update_progress, args=(step_flag, ), daemon=True).start()

    def start_loading_in_background(self, stop_flag: threading.Event):
        """
        Starts the loading update in a separate thread.
        """
        threading.Thread(target=self.update_loading, args=(stop_flag,), daemon=True).start()


import threading
import time

import keyboard
import numpy as np
import pyautogui
import torch
import win32api
import win32con
import win32gui
from PIL import Image, ImageGrab

from test import NeuralSystem


class LivePredictor:
    def __init__(self, input_size, hidden_sizes, output_size, liquid_state_hidden_size, liquid_state_num_layers,
                 loss_fn='mse'):
        self.neural_system = NeuralSystem(input_size, hidden_sizes, output_size, liquid_state_hidden_size,
                                          liquid_state_num_layers, loss_fn='mae')
        self.input_size = input_size
        self.output_size = output_size
        self.running = False
        self.training = False
        self.last_action = None
        self.current_action = None
        self.overlay = None

    def start(self, train=True):
        self.running = True
        self.training = train
        threading.Thread(target=self._run_capture).start()
        threading.Thread(target=self._run_prediction).start()
        if train:
            threading.Thread(target=self._run_training).start()

    def stop(self):
        self.running = False

    def _run_capture(self):
        while self.running:
            try:
                screenshot = np.array(ImageGrab.grab())
                mouse_pos = pyautogui.position()
                keyboard_input = keyboard.get_hotkey_name()

                # Preprocess the data to match the neural system's input size
                processed_screenshot = self._preprocess_image(screenshot)
                processed_input = np.concatenate([processed_screenshot,
                                                  np.array(mouse_pos),
                                                  self._encode_keyboard(keyboard_input)])

                self.current_action = {
                    'input': processed_input,
                    'raw_screenshot': screenshot,
                    'mouse_pos': mouse_pos,
                    'keyboard_input': keyboard_input,
                    'timestamp': time.time()
                }

                time.sleep(0.1)  # Adjust capture rate as needed
            except Exception as e:
                print(f"Error in capture: {e}")
                time.sleep(1)  # Wait before retrying

    def _run_prediction(self):
        while self.running:
            if self.current_action:
                try:
                    input_tensor = torch.tensor(self.current_action['input'], dtype=torch.float32).unsqueeze(0)
                    predicted_output = self.neural_system(input_tensor)

                    predicted_clicks = self._decode_clicks(predicted_output)
                    predicted_text = self._decode_text(predicted_output)

                    self._visualize_predictions(predicted_clicks)
                    self._display_predicted_text(predicted_text)

                except Exception as e:
                    print(f"Error in prediction: {e}")

            time.sleep(0.05)  # Adjust prediction rate as needed

    def _run_training(self):
        while self.running and self.training:
            if self.last_action and self.current_action:
                time_diff = self.current_action['timestamp'] - self.last_action['timestamp']
                if time_diff > 0.5:  # Adjust delay as needed
                    #try:
                    x_pos = torch.tensor(self.last_action['input'], dtype=torch.float32).unsqueeze(0)
                    target = torch.tensor(self.current_action['input'], dtype=torch.float32).unsqueeze(0)

                    # Use the neural system's train_step method
                    ff_losses, ls_loss = self.neural_system.train_step(x_pos, None, target)

                    print(f"Training step completed. FF Losses: {ff_losses}, LS Loss: {ls_loss}")

                    self.last_action = self.current_action
                    self.current_action = None
                #except Exception as e:
                #    print(f"Error in training: {e}")
            elif self.last_action is None and self.current_action is not None:
                self.last_action = self.current_action

            time.sleep(0.1)  # Adjust training rate as needed

    def _preprocess_image(self, image):
        # Resize and flatten the image to match the input size
        resized_image = np.array(Image.fromarray(image).resize((64, 64)))  # Adjust size as needed
        flattened_image = resized_image.flatten()
        return flattened_image  # Leave space for mouse and keyboard input

    def _encode_keyboard(self, keyboard_input):
        # Simple encoding: use the ASCII value of the first character
        if keyboard_input:
            return [ord(keyboard_input[0])]
        return [0]

    def _decode_clicks(self, predicted_output):
        # Implement logic to interpret neural network output as click predictions
        # This is a placeholder implementation
        click_data = predicted_output[:, :2].detach().numpy()
        return click_data * np.array(pyautogui.size())

    def _decode_text(self, predicted_output):
        # Implement logic to interpret neural network output as text predictions
        # This is a placeholder implementation
        text_data = predicted_output[:, 2:].detach().numpy()
        return ''.join([chr(int(i)) for i in text_data[0] if i > 0])

    def _create_overlay(self):
        screen_width, screen_height = pyautogui.size()
        self.overlay = self._create_overlay_window(screen_width, screen_height)
        self._set_transparent()

    def _create_overlay_window(self, screen_width, screen_height):
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32gui.GetModuleHandle(None)
        wc.lpszClassName = 'MyOverlayClass'
        wc.lpfnWndProc = self._wnd_proc

        class_atom = win32gui.RegisterClass(wc)

        hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
            class_atom,
            None,
            win32con.WS_POPUP,
            0, 0, screen_width, screen_height,
            None,
            None,
            wc.hInstance,
            None
        )

        return hwnd

    def _set_transparent(self):
        win32gui.SetLayeredWindowAttributes(
            self.overlay,
            win32api.RGB(0, 0, 0),
            0,
            win32con.LWA_COLORKEY
        )
        win32gui.ShowWindow(self.overlay, win32con.SW_SHOW)

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_PAINT:
            hdc, paintStruct = win32gui.BeginPaint(hwnd)
            win32gui.EndPaint(hwnd, paintStruct)
            return 0
        elif msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def __del__(self):
        if self.overlay:
            win32gui.DestroyWindow(self.overlay)

    def _visualize_predictions(self, predicted_clicks):
        if self.training:
            pass
        # print(predicted_clicks)

    def _display_predicted_text(self, predicted_text, max_length=20):
        if self.training:
            pass
        # print(predicted_text)

import atexit


@atexit.register
def save_closing():
    predictor.stop()
    predictor.neural_system.save("models/LivePredictor")


if __name__ == "__main__":
    # Usage example:
    predictor = LivePredictor(input_size=12291, hidden_sizes=[2048, 1024, 512], output_size=12291,
                              liquid_state_hidden_size=256, liquid_state_num_layers=3)
    # predictor.neural_system = predictor.neural_system.load("models/LivePredictor")
    predictor.start(train=True)  # Start prediction and training


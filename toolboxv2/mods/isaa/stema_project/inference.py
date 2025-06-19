# inference.py (Optimized for Speed)

import os
import torch
import torchaudio
from torchvision.utils import save_image
import sounddevice as sd
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

# Lokale STEMA-Importe
from stema.config import load_config
from stema.model import STEMA_Core  # Wichtig: Das neue, korrekte Modell importieren
from stema.heads import get_all_heads
from stema.data import MultimodalPatchifier
from stema.utils import get_device, load_checkpoint, show_image
from torch.nn import functional as F


def parse_input_for_task(user_input: str) -> Tuple[str, str]:
    """Determines the task and extracts the prompt from user input."""
    user_input_lower = user_input.lower()
    # Modi für Konversation und spezifische Generierung hinzufügen
    if user_input_lower.startswith("talk:") or user_input_lower.startswith("ask:"):
        return "conversation", user_input[len(user_input.split(':', 1)[0]) + 1:].strip()
    if user_input_lower.startswith("draw:"):
        return "image", user_input[len("draw:"):].strip()
    if user_input_lower.startswith("say:"):
        return "audio", user_input[len("say:"):].strip()
    if user_input_lower.startswith("summarize video:") or user_input_lower.startswith("video:"):
        prefix_len = len("summarize video:") if user_input_lower.startswith("summarize video:") else len("video:")
        return "video_to_text", user_input[prefix_len:].strip()
    if user_input_lower.startswith("describe image:") or user_input_lower.startswith("image:"):
        prefix_len = len("describe image:") if user_input_lower.startswith("describe image:") else len("image:")
        return "image_to_text", user_input[prefix_len:].strip()

    # Standard ist Konversation/Textgenerierung
    return "conversation", user_input


@torch.no_grad()
def generate_text_efficient(
    model: STEMA_Core,
    input_patches: torch.Tensor,
    patchifier: MultimodalPatchifier,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3
) -> str:
    """
    Generiert Text Token für Token effizient, indem der interne Zustand des Modells genutzt wird.
    """
    model.eval()
    tokenizer = patchifier.tokenizer

    # 1. Prompt-Verarbeitung (einmalig)
    # Die input_patches (z.B. vom Bild) werden verarbeitet, um den initialen Zustand zu setzen.
    # Der Output hier wird ignoriert, aber die internen Zustände (LSM, etc.) sind jetzt "geprimed".
    _ = model(input_patches, task="text")

    generated_token_ids = []

    # Start-Token für die Generierung (z.B. ein leeres Token oder [SEP])
    # Für einen Chatbot ist es oft gut, mit einem [SEP] oder einem speziellen Antwort-Token zu beginnen.
    # Hier starten wir mit dem letzten verarbeiteten Token, falls es einer ist, oder einem neuen.
    # Wir nehmen an, dass die Generierung "frisch" beginnt. Wir brauchen einen Start-Input.
    # Ein einzelner <BOS> (Beginning of Sentence) Token ist ein guter Start.
    current_token_id = torch.tensor([[tokenizer.bos_token_id or tokenizer.cls_token_id]], device=patchifier.device)

    for _ in range(max_new_tokens):
        # Umwandeln des aktuellen Tokens in ein Embedding
        token_embedding = patchifier.text_embed(current_token_id)  # Shape: (1, 1, dim)

        # 2. Effiziente Inferenz (nur ein Token wird verarbeitet)
        # Das Modell nutzt seine internen Zustände aus dem vorherigen Schritt.
        output_logits = model(token_embedding, task="text")  # Shape: (1, 1, vocab_size)

        # Logits für das nächste Token (da die Sequenzlänge 1 ist, ist es das erste und einzige)
        next_token_logits = output_logits[:, -1, :]  # Shape: (1, vocab_size)

        # --- Anwendung der Sampling-Strategien (wie zuvor, aber auf die Logits des einzelnen Tokens) ---
        if repetition_penalty != 1.0 and len(generated_token_ids) > 0:
            for token_id in set(generated_token_ids[-20:]):  # Nur auf kürzliche Tokens anwenden
                next_token_logits[:, token_id] /= repetition_penalty

        if no_repeat_ngram_size > 0 and len(generated_token_ids) >= no_repeat_ngram_size - 1:
            ngram_history = [tuple(generated_token_ids[i:i + no_repeat_ngram_size]) for i in
                             range(len(generated_token_ids) - no_repeat_ngram_size + 1)]
            for token_id in range(tokenizer.vocab_size):
                potential_ngram = tuple(generated_token_ids[-(no_repeat_ngram_size - 1):] + [token_id])
                if potential_ngram in ngram_history:
                    next_token_logits[:, token_id] = -float('Inf')

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            filtered_logits = torch.full_like(next_token_logits, -float('Inf'))
            filtered_logits.scatter_(1, top_k_indices, top_k_logits)
            next_token_logits = filtered_logits

        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Überprüfen auf End-of-Sequence-Token
        if next_token_id.item() == (tokenizer.eos_token_id or tokenizer.sep_token_id):
            break

        generated_token_ids.append(next_token_id.item())
        current_token_id = next_token_id  # Nächstes Token wird zum Input für die nächste Runde

    return tokenizer.decode(generated_token_ids, skip_special_tokens=True)


def run_inference(config_path: str = 'configs/default_config.yaml', checkpoint_path: Optional[str] = None) -> None:
    config = load_config(config_path)
    device = get_device(config)

    # 1. Modellkomponenten laden (mit STEMA_Core)
    patchifier = MultimodalPatchifier(config, device).to(device)
    all_heads = get_all_heads(config, patchifier).to(device)
    model = STEMA_Core(config, all_heads).to(device)
    model.eval()

    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    else:
        print("Warning: Running inference without a loaded checkpoint.")

    # Audio-Setup (unverändert)
    # ... (Code für inverse_mel_scale und griffin_lim hier einfügen, aus Platzgründen gekürzt)
    # ...

    print("\n--- STEMA-AL Interactive Inference CLI ---")
    print("Prefixes: 'draw:', 'say:', 'video:', 'image:', 'ask:'")
    print("Enter prompt or command. Type 'quit' to exit, 'reset' to clear model memory.")

    # Der Zustand wird jetzt pro Konversation gehalten, nicht global pro Sitzung.
    # `reset` löscht den aktuellen Gesprächskontext.
    model.reset_all_states()
    print("Model memory is fresh. Start a new conversation.")

    while True:
        try:
            user_input_str = input("You >> ")
        except EOFError:
            print("\nExiting.")
            break
        if not user_input_str: continue

        if user_input_str.lower() == 'quit':
            print("Exiting.")
            break
        if user_input_str.lower() == 'reset':
            model.reset_all_states()
            print("Model memory has been reset. A new conversation can begin.")
            continue

        task_type, prompt_content = parse_input_for_task(user_input_str)
        print(f"Task: {task_type.upper()}. Processing...")

        input_patches: Optional[torch.Tensor] = None
        output_task_for_model: str = task_type
        is_continuation = True  # Flag, um zu wissen, ob der Zustand zurückgesetzt werden soll

        # --- Eingabe verarbeiten ---
        if task_type in ["video_to_text", "image_to_text", "image", "audio", "video"]:
            model.reset_all_states()  # Bei neuem Medien-Input, Kontext zurücksetzen
            is_continuation = False
            print("New media input detected. Resetting conversation context.")

        if task_type == "video_to_text":
            if os.path.exists(prompt_content):
                input_patches = patchifier.process_video(prompt_content)
                output_task_for_model = "text"  # Ziel ist Textgenerierung
            else:
                print(f"File not found: {prompt_content}");
                continue
        elif task_type == "image_to_text" or task_type == "image":
            if os.path.exists(prompt_content):
                try:
                    img = Image.open(prompt_content).convert("RGB")
                    input_patches = patchifier.process_image(img)
                    output_task_for_model = "text" if task_type == "image_to_text" else "image"
                except Exception as e:
                    print(f"Error processing image: {e}");
                    continue
            else:  # Wenn kein Pfad, dann ist es ein Text-Prompt für Bildgenerierung
                input_patches = patchifier.process_text(prompt_content)
                output_task_for_model = "image"
        elif task_type == "audio":
            input_patches = patchifier.process_text(prompt_content)
        # Konversations-Modus
        elif task_type == "conversation":
            # Wenn es der Anfang einer Konversation ist (d.h. nach einem reset)
            if not is_continuation:
                model.reset_all_states()
            input_patches = patchifier.process_text(prompt_content)
            output_task_for_model = "text"
        else:  # Standard-Text-Prompt
            input_patches = patchifier.process_text(prompt_content)
            output_task_for_model = "text"

        if input_patches is None or input_patches.numel() == 0:
            print("Could not process input.");
            continue

        # --- Modellausführung und Ausgabebehandlung ---
        if output_task_for_model == "text":
            # Die alte generate_text_advanced wird durch die neue, effiziente Version ersetzt.
            # Der Prompt wird in generate_text_efficient verarbeitet, um den Zustand zu initialisieren.
            print("STEMA-AL is thinking...")
            response_text = generate_text_efficient(
                model, input_patches, patchifier,
                max_new_tokens=config['inference'].get('text_max_new_tokens', 128),
                temperature=config['inference'].get('text_temperature', 0.7),
                top_k=config['inference'].get('text_top_k', 50),
                repetition_penalty=config['inference'].get('text_repetition_penalty', 1.2),
                no_repeat_ngram_size=config['inference'].get('text_no_repeat_ngram_size', 3)
            )
            print(f"STEMA-AL >> {response_text}")

        elif output_task_for_model == "image":
            print("STEMA-AL is drawing...")
            # Für nicht-autoregressive Aufgaben wird der Zustand vor der Generierung gesetzt
            # und dann die Generierung in einem Schritt durchgeführt.
            _ = model(input_patches, task="image")  # Den Zustand mit dem Prompt setzen

            # Da ImageHead nicht autoregressiv ist, können wir den Output direkt nehmen.
            # Wir füttern einen leeren Input, um den Head mit dem gesetzten Zustand zu triggern.
            dummy_input = torch.zeros(1, 1, model.dim, device=device)
            output_from_model = model(dummy_input, task="image")

            save_path = "generated_image.png"
            save_image(output_from_model.squeeze(0).cpu(), save_path)
            print(f"STEMA-AL: Generated image saved to '{save_path}'")
            try:
                show_image(save_path)
            except Exception as e:
                print(f"Could not display image: {e}")

        elif output_task_for_model == "audio":
            print("STEMA-AL is generating audio...")
            # Ähnlich wie bei Bildern, ist die Audiogenerierung nicht zwingend autoregressiv
            # auf Token-Ebene.
            output_from_model = model(input_patches, task="audio")

            # ... (Rest der Audio-Verarbeitung, unverändert) ...
            mel_spec_output = output_from_model.squeeze(0).cpu().to(device)
            # ... (der Code für griffin_lim, etc. bleibt hier)
            print("Audio generation logic needs to be fully integrated here.")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Beispielaufruf für das Skript
    # sys.argv += ['--checkpoint', 'path/to/your/checkpoint.pth']
    run_inference()

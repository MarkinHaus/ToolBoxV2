import argparse
import os
import sys
from stema.config import load_config, get_default_config  # For ensuring default config exists
from inference import run_inference
import yaml  # For writing default config


def main():
    parser = argparse.ArgumentParser(description="STEMA: Streaming Ternary-Encoded Multimodal Architecture")
    parser.add_argument('mode', choices=['train', 'infer', 'train_rl'],
                        help="Mode: 'train' for standard training, 'infer' for inference, 'train_rl' for RL fine-tuning (conceptual).")
    parser.add_argument('--config', default='configs/default_config.yaml',
                        help="Path to the configuration file.")
    parser.add_argument('--checkpoint', default=None,
                        help="Path to a model checkpoint for inference or resuming training.")

    # Example for adding specific training URLs or inference prompts directly (optional)
    # parser.add_argument('--train_urls', nargs='+', help="List of YouTube URLs for training (overrides config).")
    # parser.add_argument('--prompt', help="Prompt for inference mode.")

    # Remove hardcoded sys.argv from original code
    # sys.argv += ['train']
    sys.argv += ['infer']
    sys.argv += ['--checkpoint', 'checkpoints/stema_pretrained_epoch_20.pth'] # Example checkpoint
    args = parser.parse_args()

    # Ensure a default config file exists if not specified or found
    if not os.path.exists(args.config) and args.config == 'configs/default_config.yaml':
        print(f"Default config not found at {args.config}. Generating one...")
        os.makedirs('configs', exist_ok=True)
        default_cfg_data = get_default_config()
        with open(args.config, 'w') as f:
            yaml.dump(default_cfg_data, f, sort_keys=False)
        print(f"Generated default config at {args.config}. Please review and customize if needed.")
        config = default_cfg_data
    else:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"Error: Config file not found at {args.config}. Exiting.")
            sys.exit(1)

    # Create necessary directories from config if they don't exist
    for path_key in ['checkpoints', 'temp_data', 'log_dir']:
        if path_key in config.get('paths', {}):
            os.makedirs(config['paths'][path_key], exist_ok=True)

    if args.mode == 'train':
        print("Starting STEMA training...")
        train_reasoning_engine(config, resume_checkpoint_path=args.checkpoint)
    elif args.mode == 'train_rl':
        print("Starting STEMA Reinforcement Learning (conceptual)...")
        # Ensure 'rl' config section exists or provide defaults
        if 'rl' not in config:
            config['rl'] = {'learning_rate': 1e-5, 'num_episodes': 100, 'save_interval': 10}  # Example defaults
        train_with_reinforcement_learning(config)  # This is a placeholder
    elif args.mode == 'infer':
        print("Starting STEMA inference...")
        if not args.checkpoint:
            print(
                "Warning: No checkpoint provided for inference. Model will use initial random weights unless a default is hardcoded.")
            # You might want to enforce checkpoint for inference or have a default one.
        run_inference(config_path=args.config, checkpoint_path=args.checkpoint)


if __name__ == '__main__':
    from toolboxv2.mods.isaa.stema_project.pretrain_decoder import main as d_main
    d_main()

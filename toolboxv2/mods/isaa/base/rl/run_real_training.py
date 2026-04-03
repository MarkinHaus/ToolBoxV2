

def main():
    """Main entry point for training."""
    from toolboxv2.mods.isaa.base.rl import TrainingPipeline

    pipeline = TrainingPipeline(
        agent_name="isaa",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        method="grpo"
    )
    results = pipeline.run_full_pipeline(deploy_ollama=True)
    print(results)


if __name__ == "__main__":
    main()


from test import NeuralSystem

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F


    def generate_dummy_data(modality, batch_size=32):
        if modality == 'audio':
            return torch.randn(batch_size, 1000)  # Flach: (batch_size, time_steps)
        elif modality == 'image':
            return torch.randn(batch_size, 3 * 224 * 224)  # Flach: (batch_size, channels * height * width)
        elif modality == 'video':
            return torch.randn(batch_size,
                               30 * 3 * 224 * 224)  # Flach: (batch_size, frames * channels * height * width)
        else:
            raise ValueError(f"Unsupported modality: {modality}")


    def resize_input(x, target_size):
        if x.shape[1] < target_size:
            return F.pad(x, (0, target_size - x.shape[1]))
        elif x.shape[1] > target_size:
            return x[:, :target_size]
        return x

    def multimodal_training_test(model, num_epochs=10, batch_size=32):
        modalities = ['audio', 'image', 'video']

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for modality in modalities:
                print(f"Training on {modality} data")

                # Generate dummy data
                x_pos = generate_dummy_data(modality, batch_size)
                x_neg = generate_dummy_data(modality, batch_size)

                # Resize inputs to match model's input size
                x_pos = resize_input(x_pos, model.input_size)
                x_neg = resize_input(x_neg, model.input_size)

                # Generate dummy targets (for simplicity, we'll use random binary targets)
                targets = torch.randint(0, 2, (batch_size, model.output_size)).float()

                # Train step
                ff_losses, ls_loss = model.train_step(x_pos, x_neg, targets)

                print(f"  Forward-Forward Losses: {ff_losses}")
                print(f"  Liquid State Loss: {ls_loss}")

            print("Testing inference:")
            for modality in modalities:
                test_input = generate_dummy_data(modality, 1)
                test_input = resize_input(test_input, model.input_size)
                output = model(test_input)
                print(f"  {modality.capitalize()} output shape: {output.shape}")

            print("---")


    # Annahme: Sie haben bereits ein NeuralSystem-Objekt erstellt
    input_size = 30 * 3 * 224 * 224  # Größte Eingabedimension (Video)
    hidden_sizes = [512, 256, 128]
    output_size = 10
    liquid_state_hidden_size = 64
    liquid_state_num_layers = 3

    model = NeuralSystem(input_size, hidden_sizes, output_size,
                         liquid_state_hidden_size, liquid_state_num_layers)

    # Führen Sie den multimodalen Trainingstest durch
    multimodal_training_test(model)

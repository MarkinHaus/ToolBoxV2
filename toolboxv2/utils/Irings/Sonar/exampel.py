# example_usage.py
import asyncio
import numpy as np
from sdk import ASonarClient, SonarConfig, SonarClient


async def amain():
    config = SonarConfig(
        api_key="your_api_key",
        base_url="http://localhost:8080"
    )

    async with ASonarClient(config) as client:
        # Text embeddings (batch)
        texts = ["Hello world!", "How are you?", "Natural language processing"]
        embeddings = await client.text_to_embeddings(texts, source_lang="eng_Latn")
        print(f"Embeddings shape: {embeddings.shape}")

        # Text embeddings (streaming)
        async for batch in await client.text_to_embeddings(texts, stream=True):
            print(f"Progress: {batch['progress']*100:.1f}%")
            print(f"Batch result shape: {np.array(batch['batch_result']).shape}")

        # Translation
        translations = await client.translate_text(
            texts=["Hello, how are you?"],
            source_lang="eng_Latn",
            target_lang="fra_Latn"
        )
        print(f"Translation: {translations[0]}")

        # Speech processing
        speech_files = ["audio1.wav", "audio2.wav"]
        transcriptions = await client.speech_to_text(speech_files)
        print(f"Transcriptions: {transcriptions}")

        # Similarity scoring
        scores = await client.compute_similarity(
            source_texts=["This is a test."],
            translated_texts=["C'est un test."],
            source_lang="eng_Latn",
            target_lang="fra_Latn"
        )
        print(f"Similarity score: {scores[0]}")


def main():
    config = SonarConfig(
        api_key="your_api_key",
        base_url="http://localhost:8080"
    )

    with SonarClient(config) as client:
        # Text embeddings (batch)
        texts = ["Hello world!", "How are you?", "Natural language processing"]
        embeddings = client.text_to_embeddings(texts, source_lang="eng_Latn")
        print(f"Embeddings shape: {embeddings.shape}")

        # Text embeddings (streaming)
        for batch in client.text_to_embeddings(texts, stream=True):
            print(f"Progress: {batch['progress']*100:.1f}%")
            print(f"Batch result shape: {np.array(batch['batch_result']).shape}")

        # Translation
        translations = client.translate_text(
            texts=["Hello, how are you?"],
            source_lang="eng_Latn",
            target_lang="fra_Latn"
        )
        print(f"Translation: {translations[0]}")

        # Speech processing
        speech_files = ["audio1.wav", "audio2.wav"]
        transcriptions = client.speech_to_text(speech_files)
        print(f"Transcriptions: {transcriptions}")

        # Similarity scoring
        scores = client.compute_similarity(
            source_texts=["This is a test."],
            translated_texts=["C'est un test."],
            source_lang="eng_Latn",
            target_lang="fra_Latn"
        )
        print(f"Similarity score: {scores[0]}")


if __name__ == "__main__":
    main()
    asyncio.run(amain())

# %pip install --upgrade --quiet  yt_dlp
# %pip install --upgrade --quiet  pydub
# %pip install --upgrade --quiet  librosa
# %pip install --upgrade --quiet  youtube-transcript-api
# %pip install --upgrade --quiet  pytube
# YouTube url to text

# Use YoutubeAudioLoader to fetch / download the audio files.

# Then, ues OpenAIWhisperParser() to transcribe them to text.

# Let’s take the first lecture of Andrej Karpathy’s YouTube course as an example!
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat


def get_videos2audio_files(urls, save_dir):
    if isinstance(urls, str):
        urls = [urls]
    loader = YoutubeAudioLoader(urls=urls, save_dir=save_dir)
    blobs = lambda: loader.yield_blobs()
    return loader, blobs


def get_video2transcription(url, add_video_info=False
                            , translation="en"
                            , transcript_format=None
                            , continue_on_failure=TranscriptFormat.TEXT
                            ):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False
                                            , translation="en"
                                            , transcript_format=None
                                            , continue_on_failure=TranscriptFormat.TEXT
                                            )
    blobs = lambda: loader.load()
    return loader, blobs

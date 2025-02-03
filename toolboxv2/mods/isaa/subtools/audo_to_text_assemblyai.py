# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)
import os
try:
    from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
    from langchain_community.document_loaders.assemblyai import TranscriptFormat
    Audio_Avalabel = True
except ImportError as e:
    Audio_Avalabel = e

_api_key_ = lambda: os.environ.get('ASSEMBLYAI_API_KEY')


def translate(filename_url, transcript_format=TranscriptFormat.TEXT, api_key=_api_key_()):
    if Audio_Avalabel is not True:
        raise ImportError(e)
    loader = AssemblyAIAudioTranscriptLoader(file_path=filename_url, transcript_format=transcript_format,
                                             api_key=api_key)

    docs = lambda: loader.load()
    return loader, docs


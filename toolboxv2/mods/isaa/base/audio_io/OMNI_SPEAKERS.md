# Omni Speaker Recognition — User Story (icli)

How a real, persistent speaker identity works in Omni voice mode, and how to add
your own voice so Omni greets/addresses people by name.

## What it does

In Omni mode every finished user utterance is embedded and matched against a
persisted registry (`isaa/omni/speakers.json`). On a confident match the model
receives `[speaker: <name>]` context and a `SPEAKER_DETECTED` phase fires; on a
new/unknown voice it stays unlabeled until you enroll it.

The embedding model is **the same pyannote model** used by the classic
`/audio speaker` enrollment path, so a voice enrolled once is recognized in both
the live pipeline and Omni.

## Requirements (for REAL recognition)

Without these, Omni runs with the dependency-free **stub** embedder: it won't
crash, but it can't actually tell speakers apart.

    pip install pyannote.audio
    export HF_TOKEN=hf_...        # HuggingFace token, accept the pyannote/embedding license

At Omni start you'll see one of:

    [speakers] real embedder active (pyannote)
    [speakers] stub embedder (set HF_TOKEN + pip install pyannote.audio for real ID)

## User story: add a speaker

1. Enroll your voice (records 5 s, extracts the embedding, stores it):

       /audio speaker add <name>

   Speak normally for the countdown. The profile is saved to the speaker store.

2. Start Omni:

       /omni start

3. Speak. When your voice matches, the footer shows the detected speaker and the
   model is told `[speaker: <name>]`. Unknown voices stay unlabeled — enroll them
   the same way.

Manage profiles:

    /audio speaker list            # registered names
    /audio speaker remove <name>   # delete a profile
    /audio speaker who             # who is currently detected

## Notes

- pcm fed to the embedder is int16 mono @16 kHz (TARGET_SR); the adapter handles
  the conversion to the model's expected waveform tensor.
- The Omni registry (`isaa/omni/speakers.json`) is separate from the live
  pipeline store, but both use the same model — enroll once per store if you want
  recognition in both. (A shared store is a future nicety, not required.)
- If the model can't load (missing token/deps), `embed()` returns `None` and Omni
  simply treats every voice as unknown — no failure in the audio loop.

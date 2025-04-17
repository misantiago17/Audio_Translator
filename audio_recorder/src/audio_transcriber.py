# audio_transcriber.py
# Requires: pip install openai-whisper (and ffmpeg)

import os
import wave
import whisper

# Load local Whisper model (CPU, float32 to avoid FP16 fallback warning)
model = whisper.load_model("base", device="cpu", compute_type="float32")


def save_frames_to_wav(frames, path, rate, channels, sample_width):
    """Save raw audio frames to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))


def transcribe_audio(file_path: str) -> str:
    """Transcribe a WAV audio file using local Whisper model."""
    result = model.transcribe(file_path)
    return result.get("text", "")


def transcribe_from_recorder(recorder, output_wav: str = "recording.wav") -> str:
    """Save recorder frames to WAV and transcribe."""
    # Determine the sample width from the recorder
    sample_width = recorder.audio.get_sample_size(recorder.fmt)
    save_frames_to_wav(
        recorder.frames,
        output_wav,
        recorder.rate,
        recorder.channels,
        sample_width
    )
    return transcribe_audio(output_wav)


if __name__ == "__main__":
    from audio_recorder import AudioRecorder

    recorder = AudioRecorder()
    input("Press Enter to start recording...")
    recorder.start_recording()
    input("Press Enter to stop recording...")
    recorder.stop_recording()

    print("Transcribing...")
    transcription = transcribe_from_recorder(recorder)
    print("Transcription:\n", transcription) 
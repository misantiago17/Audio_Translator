# Audio Translator

A real-time audio transcription and translation system that captures audio streams and converts speech to text using OpenAI's Whisper model.

## Features

- Real-time audio capture using PyAudio
- Stream-based speech recognition using Whisper
- Configurable transcription settings (language, model size)
- Callback system for real-time updates
- Efficient memory and CPU usage

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/audio_translator.git
cd audio_translator
```

2. Create a virtual environment and activate it:
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- PyAudio
- NumPy
- OpenAI Whisper
- PyTorch
- Threading

## Usage

### Stream Recording Demo

To run the real-time transcription demo:

```bash
python audio_recorder/demo_stream.py --model tiny --language en --duration 60
```

Parameters:
- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--language`: Language code for transcription (en, pt, es, etc.)
- `--device`: Audio device index (optional)
- `--duration`: Recording duration in seconds (default: 30)

### Using the StreamRecorder in Your Code

```python
from audio_recorder.src.whisper_transcriber import WhisperTranscriber
from audio_recorder.src.stream_recorder import StreamRecorder

# Initialize transcriber
transcriber = WhisperTranscriber(
    model_size="tiny",
    language="en",
    device="cuda" if WhisperTranscriber.is_cuda_available() else "cpu"
)

# Initialize recorder
recorder = StreamRecorder(
    transcriber=transcriber,
    device_index=None,  # Auto-select microphone
    callback_interval=0.5  # Update every 0.5 seconds
)

# Add callback for real-time updates
def on_transcription(text):
    print(f"Transcription: {text}")

recorder.add_transcription_callback(on_transcription)

# Start recording
recorder.start()

# Record for 10 seconds
import time
time.sleep(10)

# Stop recording
recorder.stop()

# Clean up
del recorder
del transcriber
```

## Project Structure

```
audio_translator/
├── audio_recorder/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── stream_recorder.py  # Real-time audio capture
│   │   └── whisper_transcriber.py  # Speech-to-text using Whisper
│   └── demo_stream.py  # Demo application
├── requirements.txt
└── README.md
```

## License

MIT

## Credits

This project uses OpenAI's Whisper model for speech recognition. 
# constants.py - Arquivo central para configurações compartilhadas
import pyaudio

# Configurações de áudio
WINDOW_SECONDS = 15        # Tamanho da janela de buffer em segundos
SAMPLE_RATE = 16000        # Taxa de amostragem (16kHz - ideal para Whisper)
CHUNK_SIZE = 1024          # Tamanho de cada fragmento de áudio
CHANNELS = 1               # Mono (1 canal - ideal para Whisper)
SAMPLE_FORMAT = pyaudio.paInt16  # Linear 16-bit PCM (ideal para Whisper)

# Configurações de transcrição
WHISPER_MODEL = "medium"   # Modelo do Whisper (tiny, base, small, medium, large)
SEGMENT_LENGTH = 10        # Tamanho do segmento de áudio em segundos
HOP_LENGTH = 5             # Tempo entre processamentos consecutivos (segundos)

# Configurações de sistema de arquivos
TEMP_DIR = "temp"          # Diretório para arquivos temporários
DEFAULT_OUTPUT_WAV = f"{TEMP_DIR}/recording.wav"  # Nome padrão do arquivo de saída 
# constants.py - Arquivo central para configurações compartilhadas
import pyaudio

# Configurações de áudio
WINDOW_SECONDS = 10        # Tamanho da janela de buffer em segundos (reduzido)
SAMPLE_RATE = 16000        # Taxa de amostragem (16kHz - ideal para Whisper)
CHUNK_SIZE = 1024          # Tamanho de cada fragmento de áudio
CHANNELS = 1               # Mono (1 canal - ideal para Whisper)
SAMPLE_FORMAT = pyaudio.paInt16  # Linear 16-bit PCM (ideal para Whisper)

# Configurações de transcrição
WHISPER_MODEL = "base"     # Modelo do Whisper (tiny, base, small, medium, large)
                           # "tiny" (39M) ou "base" (74M) consomem muito menos recursos que "medium" (769M)
SEGMENT_LENGTH = 5         # Tamanho do segmento de áudio em segundos (reduzido)
HOP_LENGTH = 2.5           # Tempo entre processamentos consecutivos (reduzido)

# Configurações de sistema de arquivos
TEMP_DIR = "temp"          # Diretório para arquivos temporários
DEFAULT_OUTPUT_WAV = f"{TEMP_DIR}/recording.wav"  # Nome padrão do arquivo de saída

# Configurações do modelo Whisper
DEVICE_TYPE = "cpu"        # Dispositivo para processamento ("cpu" ou "cuda" para GPU)
TEMPERATURE = 0.0          # Temperatura para aleatoriedade na geração de texto (0.0 = determinístico)

# Configurações de otimização de desempenho
LIMIT_HISTORY = True       # Limitar histórico para economizar memória
MAX_HISTORY_SECONDS = 10   # Máximo de segundos de áudio a manter no histórico 
import pyaudio
import threading
import collections
import sys
import os

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    WINDOW_SECONDS, 
    SAMPLE_RATE, 
    CHUNK_SIZE, 
    CHANNELS, 
    SAMPLE_FORMAT
)

class AudioRecorder:
    def __init__(self, window_seconds=WINDOW_SECONDS):
        # Inicializa a interface PyAudio
        p = pyaudio.PyAudio()
        
        # Tenta encontrar o dispositivo de cabo virtual (VB-Cable)
        # Um cabo virtual permite capturar o áudio do sistema em vez do microfone
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            # Procura dispositivos com 'cable' no nome (como VB-Cable)
            if info['maxInputChannels'] > 0 and 'cable' in info['name'].lower():
                device_index = i
                break
                
        # Se não encontrar cabo virtual, usa o dispositivo de entrada padrão
        if device_index is None:
            device_index = p.get_default_input_device_info()['index']
            
        # Configuração dos parâmetros de gravação
        self.input_device_index = device_index  # Índice do dispositivo a usar
        self.audio = p  # Objeto PyAudio
        self.recording = False  # Estado da gravação
        self.rate = SAMPLE_RATE  # Taxa de amostragem
        self.chunk = CHUNK_SIZE  # Tamanho de cada fragmento de áudio
        self.channels = CHANNELS  # Número de canais
        self.fmt = SAMPLE_FORMAT  # Formato de áudio
        
        # Calcula quantos chunks cabem na janela de tempo desejada
        self.max_chunks = int((self.rate * window_seconds) / self.chunk)
        # Usa deque como buffer circular com tamanho máximo
        self.frames = collections.deque(maxlen=self.max_chunks)
        
        # Lock para acesso seguro ao buffer em operações de leitura
        self.buffer_lock = threading.Lock()

    def start_recording(self):
        # Evita iniciar múltiplas gravações
        if self.recording:
            return
            
        # Limpa o buffer e marca como gravando
        with self.buffer_lock:
            self.frames.clear()
        self.recording = True
        
        # Inicia a gravação em uma thread separada para não bloquear a GUI
        # daemon=True faz a thread parar quando o programa principal termina
        threading.Thread(target=self._record, daemon=True).start()

    def _record(self):
        # Abre um stream de áudio para capturar o som
        stream = self.audio.open(format=self.fmt,
                               channels=self.channels,
                               rate=self.rate,
                               input=True,
                               input_device_index=self.input_device_index,
                               frames_per_buffer=self.chunk)
                               
        # Continua gravando enquanto self.recording for True
        while self.recording:
            # Lê um chunk de áudio e adiciona ao buffer circular
            # exception_on_overflow=False evita erros se o buffer estiver cheio
            chunk = stream.read(self.chunk, exception_on_overflow=False)
            with self.buffer_lock:
                self.frames.append(chunk)
            
        # Finaliza o stream quando a gravação termina
        stream.stop_stream()
        stream.close()

    def stop_recording(self):
        # Define recording como False, fazendo o loop em _record parar
        self.recording = False

    def get_recording_status(self):
        # Retorna se está gravando ou não
        return self.recording
    
    def get_frames(self):
        # Retorna uma cópia segura dos frames atuais
        with self.buffer_lock:
            return list(self.frames)

    def cleanup(self):
        # Libera recursos do PyAudio ao fechar o programa
        self.audio.terminate()
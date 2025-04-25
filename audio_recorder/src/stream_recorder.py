import time
import threading
import numpy as np
import pyaudio
import queue
import logging
import gc
import traceback
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
import datetime

from .whisper_transcriber import WhisperTranscriber

# Configurações globais
SAMPLE_RATE = 16000  # Taxa de amostragem compatível com Whisper
CHUNK_SIZE = 1024    # Tamanho do chunk de áudio
CHANNELS = 1         # Mono
FORMAT = pyaudio.paInt16  # Formato de áudio
MAX_HISTORY_SECONDS = 30  # Máximo de história de áudio em segundos

# Configurações para detecção de silêncio
SILENCE_THRESHOLD = 0.005  # Limiar para considerar como silêncio (amplitude)
MIN_SILENCE_DURATION = 0.7  # Duração mínima de silêncio para considerar pausa natural (segundos)
MAX_SPEECH_DURATION = 7.0   # Duração máxima de fala contínua sem pausas forçadas (segundos)

# Configure logger
logger = logging.getLogger(__name__)

class StreamRecorder:
    """
    Classe para capturar áudio em tempo real e transcrever usando WhisperTranscriber.
    Gerencia entrada de áudio e processamento contínuo.
    """
    
    def __init__(self, 
                 transcriber,
                 on_transcription: Optional[Callable[[str], None]] = None,
                 device_index: Optional[int] = None,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1):
        """
        Inicializa o gravador de stream.
        
        Args:
            transcriber: Instância do WhisperTranscriber para transcrição
            on_transcription: Callback para quando uma transcrição estiver disponível
            device_index: Índice do dispositivo de entrada (None = padrão)
            sample_rate: Taxa de amostragem (Hz)
            chunk_size: Tamanho de cada chunk de áudio
            channels: Número de canais (1=mono, 2=estéreo)
        """
        self.transcriber = transcriber
        self.on_transcription = on_transcription
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        # Inicializa PyAudio
        try:
            self._pyaudio = pyaudio.PyAudio()
        except Exception as e:
            logger.error(f"Erro ao inicializar PyAudio: {str(e)}")
            self._pyaudio = None
        
        # Estado interno
        self._stream = None
        self._stop_event = threading.Event()
        self._process_thread = None
        
        # Buffer de acumulação de áudio
        self._audio_buffer_lock = threading.Lock()
        self._accumulated_buffer = np.array([], dtype=np.float32)
        
        # Estado para detecção de silêncio
        self._is_speech = False
        self._silence_start = 0
        self._speech_start = 0
        self._last_process_time = time.time()
        
    def _detect_silence(self, audio_data):
        """
        Detecta silêncio no áudio para identificar pausas naturais na fala.
        
        Args:
            audio_data: Array numpy com dados de áudio
            
        Returns:
            bool: True se for silêncio, False se for fala
        """
        if audio_data is None or len(audio_data) == 0:
            return True
            
        # Calcula amplitude RMS do áudio
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return rms < SILENCE_THRESHOLD
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback chamado pelo PyAudio quando novo áudio está disponível.
        
        Args:
            in_data: Bytes de áudio do microfone
            frame_count: Número de frames recebidos
            time_info: Informações de tempo
            status: Status da captura
        
        Returns:
            Tuple: (None, pyaudio.paContinue)
        """
        if self._stop_event.is_set():
            return None, pyaudio.paComplete
            
        try:
            # Converte os bytes para float32 para processamento de alta qualidade
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Adiciona ao buffer acumulado com sincronização
            with self._audio_buffer_lock:
                self._accumulated_buffer = np.append(self._accumulated_buffer, audio_data)
                
            # Atualiza o estado de silêncio/fala
            is_silence = self._detect_silence(audio_data)
            current_time = time.time()
            
            # Se mudamos de fala para silêncio, marca o início do silêncio
            if not is_silence and not self._is_speech:
                self._is_speech = True
                self._speech_start = current_time
                
            # Se mudamos de silêncio para fala, marca o fim do silêncio
            elif is_silence and self._is_speech:
                self._is_speech = False
                self._silence_start = current_time
                
        except Exception as e:
            logger.error(f"Erro no callback de áudio: {str(e)}")
            
        return None, pyaudio.paContinue
    
    def _process_audio_loop(self):
        """Thread para processar o áudio acumulado regularmente
        e enviar para transcrição."""

        last_position = 0
        min_audio_length = 1.5  # segundos (reduzido para maior responsividade)
        min_samples = int(min_audio_length * self.sample_rate)
        force_process_interval = 5.0  # segundos
        last_process_time = time.time()
        
        # Aumenta o tamanho do buffer para manter mais contexto
        max_buffer_size = int(60 * self.sample_rate)  # 60 segundos de áudio
        overlap_samples = int(2.0 * self.sample_rate)  # 2 segundos de sobreposição

        while not self._stop_event.is_set():
            # Copia dados atuais do buffer para processamento
            with self._audio_buffer_lock:
                if len(self._accumulated_buffer) == 0:
                    time.sleep(0.1)
                    continue
                    
                audio_buffer = np.copy(self._accumulated_buffer)
                current_position = len(audio_buffer)
                
            # Calcula quanto áudio novo temos desde o último processamento
            new_samples = current_position - last_position
            current_time = time.time()
            time_since_last_process = current_time - last_process_time
            
            # Verificação de pausas naturais na fala
            should_process = False
            
            # Caso 1: Temos uma pausa natural na fala (silêncio após fala)
            silence_duration = 0
            if not self._is_speech and self._silence_start > 0:
                silence_duration = current_time - self._silence_start
                # Se encontramos uma pausa natural significativa
                if silence_duration >= MIN_SILENCE_DURATION and new_samples >= min_samples:
                    should_process = True
                    logger.debug(f"Processando devido a pausa natural de {silence_duration:.2f}s")
            
            # Caso 2: Fala contínua por tempo prolongado
            speech_duration = 0
            if self._is_speech and self._speech_start > 0:
                speech_duration = current_time - self._speech_start
                if speech_duration >= MAX_SPEECH_DURATION and new_samples >= min_samples:
                    should_process = True
                    logger.debug(f"Processando devido a fala contínua de {speech_duration:.2f}s")
            
            # Caso 3: Tempo mínimo passado e temos áudio suficiente
            sufficient_audio = new_samples >= min_samples
            time_threshold = 2.0 if self._is_speech else 4.0  # Processa mais frequentemente durante fala
            if time_since_last_process >= time_threshold and sufficient_audio:
                should_process = True
                
            # Caso 4: Força o processamento após tempo máximo
            if time_since_last_process >= force_process_interval and new_samples > 0:
                should_process = True
                logger.debug("Processamento forçado por tempo limite")
            
            if should_process:
                # Limita o tamanho do buffer, mas mantém sobreposição
                if len(audio_buffer) > max_buffer_size:
                    # Mantém os últimos N segundos + sobreposição para contexto
                    audio_buffer = audio_buffer[-(max_buffer_size):]
                
                # Requisita transcrição do áudio atual (com contexto acumulado)
                _, transcription = self.transcriber.transcribe_stream(
                    audio_buffer, 
                    accumulate=True  # Garante que o contexto seja mantido
                )
                
                last_position = current_position
                last_process_time = time.time()
                
                # Processa o resultado da transcrição
                if transcription and self.on_transcription:
                    try:
                        self.on_transcription(transcription)
                    except Exception as e:
                        logger.error(f"Erro no callback de transcrição: {str(e)}")
                        
                # Reseta contadores de silêncio/fala após processamento
                if not self._is_speech:
                    self._silence_start = 0
                if self._is_speech:
                    self._speech_start = time.time()  # Reinicia o contador de fala
            
            time.sleep(0.05)  # Reduzido para maior responsividade
    
    def start(self):
        """
        Inicia a gravação e processamento do áudio em streaming.
        Configura o PyAudio e inicia a captura de áudio.
        """
        if self._stream is not None:
            logger.warning("Gravação já está em andamento")
            return False
            
        try:
            # Reinicializa PyAudio se necessário
            if self._pyaudio is None:
                self._pyaudio = pyaudio.PyAudio()
            
            # Configura e abre o stream de áudio
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
                stream_callback=self._audio_callback
            )
            
            # Prepara para iniciar
            self._stop_event.clear()
            
            # Inicializa buffer vazio
            with self._audio_buffer_lock:
                self._accumulated_buffer = np.array([], dtype=np.float32)
            
            # Limpa contexto de transcrição para nova sessão
            self.transcriber.clear_stream_context()
            
            # Inicializa estado de silêncio
            self._is_speech = False
            self._silence_start = 0
            self._speech_start = 0
            self._last_process_time = time.time()
            
            # Inicia thread de processamento
            self._process_thread = threading.Thread(
                target=self._process_audio_loop,
                daemon=True
            )
            self._process_thread.start()
            
            # Inicia o stream
            self._stream.start_stream()
            
            logger.info("Gravação iniciada com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar gravação: {str(e)}")
            self.stop()  # Limpa recursos em caso de erro
            return False
            
    def stop(self):
        """
        Para a gravação e libera recursos.
        """
        # Sinaliza para threads pararem
        self._stop_event.set()
        
        # Fecha o stream de áudio
        if self._stream is not None:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            
        # Finaliza PyAudio
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
            
        # Aguarda thread de processamento terminar (com timeout)
        if self._process_thread is not None and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
            
        logger.info("Gravação finalizada")
        return True
    
    def __del__(self):
        """Limpeza de recursos ao destruir a instância"""
        self.stop()
        if self._pyaudio:
            self._pyaudio.terminate() 
import pyaudio
import threading
import collections
import sys
import os
import time
import wave
import numpy as np
import logging
from datetime import datetime

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    WINDOW_SECONDS, 
    SAMPLE_RATE, 
    CHUNK_SIZE, 
    CHANNELS, 
    SAMPLE_FORMAT
)

from transcription_base import transcribe_audio, save_frames_to_wav

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio_recorder")

class AudioRecorder:
    def __init__(self, window_seconds=WINDOW_SECONDS, output_dir="recordings"):
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
        
        # Variáveis para transcrição em tempo real
        self.realtime_transcription = False
        self.chunk_duration = 2.0  # Em segundos
        self.chunk_frames = []
        self.transcription_callback = None
        self.last_chunk_time = 0
        
        # Função de processamento para cada chunk coletado
        self.chunk_processor = None
        
        # Diretório de saída para arquivos gravados
        self.output_dir = output_dir
        
        # Cria o diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)

    def start_recording(self):
        # Evita iniciar múltiplas gravações
        if self.recording:
            logger.warning("Gravação já está em andamento")
            return
            
        # Limpa o buffer e marca como gravando
        with self.buffer_lock:
            self.frames.clear()
        self.recording = True
        
        # Inicializa ou reinicializa PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Abre o stream de áudio
        self.stream = self.audio.open(
            format=self.fmt,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk
        )
        
        self.chunk_frames = []
        self.last_chunk_time = time.time()
        
        # Inicia thread para gravação em background
        self.thread = threading.Thread(target=self._record_thread, daemon=True)
        self.thread.start()
        
        logger.info("Gravação iniciada")

    def _record_thread(self):
        """Função executada em thread para gravação contínua."""
        try:
            # Contador para limitar a frequência de processamento em paralelo
            frame_counter = 0
            
            # Tempo da última limpeza de memória
            last_cleanup_time = time.time()
            
            while self.recording:
                # Lê dados do stream
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                except OSError as e:
                    # Erro comum quando o dispositivo de áudio é alterado ou desconectado
                    logger.error(f"Erro ao ler do stream de áudio: {e}")
                    time.sleep(0.1)  # Pausa breve para evitar loop de erro intensivo
                    continue
                
                # Converte para array numpy para processamento
                frame_data = np.frombuffer(data, dtype=np.int16)
                
                # Adiciona ao buffer de frames
                with self.buffer_lock:
                    self.frames.append(frame_data)
                
                # Adiciona ao buffer de chunks somente se transcrição em tempo real estiver ativa
                if self.realtime_transcription:
                    self.chunk_frames.append(frame_data)
                    
                    # Verifica se é hora de processar um chunk
                    current_time = time.time()
                    chunk_time = current_time - self.last_chunk_time
                    
                    if chunk_time >= self.chunk_duration:
                        self._process_audio_chunk()
                        self.last_chunk_time = current_time
                
                # Processador de chunks personalizado - executa com menos frequência
                if self.chunk_processor and frame_counter % 5 == 0:  # A cada 5 frames ao invés de cada frame
                    # Evita criar cópias desnecessárias que desperdiçam memória
                    with self.buffer_lock:
                        # Usa uma referência direta ao invés de fazer uma cópia completa
                        # Os frames individuais não serão modificados, então é seguro
                        frames_ref = list(self.frames)
                    
                    # Limita a criação de novas threads - executa diretamente a cada 10 ciclos
                    if frame_counter % 10 == 0:
                        self.chunk_processor(frames_ref, False)
                    else:
                        # Usa thread somente de vez em quando para manter responsividade
                        threading.Thread(
                            target=self.chunk_processor,
                            args=(frames_ref, False),
                            daemon=True
                        ).start()
                
                # Incrementa o contador de frames
                frame_counter += 1
                
                # A cada 60 segundos, verifica se é necessário limpar recursos
                current_time = time.time()
                if current_time - last_cleanup_time > 60:
                    # Força liberação de memória não utilizada
                    import gc
                    gc.collect()
                    last_cleanup_time = current_time
                        
        except Exception as e:
            logger.error(f"Erro durante gravação: {e}")
            self.recording = False

    def _process_audio_chunk(self):
        """Processa um chunk de áudio para transcrição em tempo real."""
        if not self.chunk_frames or not self.transcription_callback:
            return
            
        # Verifica se o buffer é grande o suficiente para processar
        if len(self.chunk_frames) < 5:  # Evita processar buffers muito pequenos
            return
            
        # Concatena frames do chunk em um único array
        chunk_data = np.concatenate(self.chunk_frames)
        self.chunk_frames = []  # Limpa o buffer do chunk
        
        # Limita o tamanho do chunk para evitar uso excessivo de memória
        max_chunk_size = 3 * 16000  # Máximo 3 segundos a 16kHz
        if len(chunk_data) > max_chunk_size:
            chunk_data = chunk_data[-max_chunk_size:]
        
        # Envia para transcrição em uma thread separada para não bloquear a gravação
        # Reutiliza o mesmo thread quando possível para não criar threads excessivos
        threading.Thread(
            target=self._transcribe_chunk, 
            args=(chunk_data,),  # Não faz cópia, usa diretamente o array
            daemon=True
        ).start()

    def _transcribe_chunk(self, chunk_data):
        """Transcreve um chunk de áudio e chama o callback com o resultado."""
        try:
            # Converte para float32 antes de transcrever
            # Normaliza para intervalo [-1.0, 1.0] que é o esperado pelo modelo Whisper
            # Faz isso in-place quando possível para economizar memória
            if chunk_data.dtype != np.float32:
                chunk_data_float = chunk_data.astype(np.float32) / 32768.0
            else:
                # Se já for float32, normaliza in-place
                chunk_data_float = chunk_data / np.max(np.abs(chunk_data)) if np.max(np.abs(chunk_data)) > 0 else chunk_data
            
            # Transcreve o chunk - passa uma visão do array ao invés de uma cópia
            result = transcribe_audio(frames=chunk_data_float, sample_rate=self.rate)
            
            # Chama o callback com o resultado
            if self.transcription_callback:
                self.transcription_callback(result, False)  # False indica que não é o texto final
                
            # Limpa referências para ajudar o GC
            del chunk_data_float
                
        except Exception as e:
            logger.error(f"Erro na transcrição de chunk: {e}")
            if self.transcription_callback:
                self.transcription_callback(f"[Erro: {str(e)}]", False)
                
    def set_realtime_transcription(self, enabled, chunk_duration=2.0, callback=None):
        """
        Configura a transcrição em tempo real.
        
        Args:
            enabled: Se True, ativa a transcrição em tempo real
            chunk_duration: Duração de cada chunk em segundos
            callback: Função a ser chamada quando um texto é transcrito
                      Assinatura: callback(texto, is_final)
        """
        # Limpa todas as variáveis relacionadas à transcrição anterior
        if not enabled and self.realtime_transcription:
            self.chunk_frames = []
            self.transcription_callback = None
            import gc
            gc.collect()  # Força liberação de memória não utilizada
        
        self.realtime_transcription = enabled
        self.chunk_duration = chunk_duration
        self.transcription_callback = callback
        self.chunk_frames = []
        self.last_chunk_time = time.time()
        
        logger.info(f"Transcrição em tempo real: {'ativada' if enabled else 'desativada'}")
        
    def stop_recording(self):
        # Define recording como False, fazendo o loop em _record parar
        self.recording = False
        
        # Chama o processador de chunk com o parâmetro final=True para finalizar
        if self.chunk_processor:
            try:
                # Obtém uma cópia segura de todos os frames
                with self.buffer_lock:
                    frames_ref = list(self.frames)
                
                # Chama o processador de forma síncrona para garantir processamento final
                self.chunk_processor(frames_ref, True)  # True indica que é o último chunk
            except Exception as e:
                logger.error(f"Erro ao processar o chunk final: {e}")
        
        # Fecha o stream e libera recursos
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        # Libera memória
        self.chunk_frames = []
        
        # Força coleta de lixo para liberar memória
        import gc
        gc.collect()
        
        logger.info("Gravação finalizada")

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

    def set_chunk_processor(self, processor_func):
        """
        Define uma função para processar chunks de áudio.
        Esta função será chamada periodicamente durante a gravação.
        
        Args:
            processor_func: Função que recebe (frames, is_final) como parâmetros
                           - frames: Lista de arrays numpy com dados de áudio
                           - is_final: Booleano que indica se é o processamento final
        """
        self.chunk_processor = processor_func
        logger.info("Processador de chunks configurado")

    def transcribe_recording(self):
        """Transcreve a gravação atual e retorna o texto."""
        if not self.frames:
            logger.warning("Não há dados de áudio para transcrever")
            return "Nenhum áudio para transcrever"
            
        # Concatena todos os frames em um único array
        all_frames = np.concatenate(self.frames)
        
        # Converte para float32 e normaliza para o intervalo [-1.0, 1.0]
        all_frames_float = all_frames.astype(np.float32) / 32768.0
        
        # Envia para transcrição
        result = transcribe_audio(frames=all_frames_float, sample_rate=self.rate)
        
        # Se transcrição em tempo real estiver ativada, marca como texto final
        if self.realtime_transcription and self.transcription_callback:
            self.transcription_callback(result, True)  # True indica que é o texto final
            
        return result
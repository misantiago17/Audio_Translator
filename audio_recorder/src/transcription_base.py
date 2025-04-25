# transcription_base.py
# Módulo base para diferentes tipos de transcritores de áudio para texto
# Define interfaces, funções comuns e funções de conveniência

import os
import wave
import abc
import sys
import librosa
import numpy as np
import logging
import time
import io
from pathlib import Path
import tempfile
from datetime import datetime

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    SEGMENT_LENGTH,    # Duração de cada segmento para processamento (segundos)
    TEMP_DIR,          # Diretório para arquivos temporários
    DEFAULT_OUTPUT_WAV # Caminho padrão para o arquivo WAV de saída
)

# Removendo a importação circular
# from whisper_transcriber import WhisperTranscriber

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("transcription_base")

# Vamos mover a inicialização do transcritor para depois da definição da classe
# default_transcriber = WhisperTranscriber()

class AudioTranscriber(abc.ABC):
    """
    Classe base abstrata para transcritores de áudio.
    
    Qualquer novo transcritor deve herdar desta classe e implementar
    o método transcribe_file.
    """
    
    @abc.abstractmethod
    def transcribe_file(self, file_path: str, initial_prompt: str = None) -> str:
        """
        Transcreve um arquivo de áudio para texto.
        
        Parâmetros:
            file_path (str): Caminho para o arquivo de áudio
            initial_prompt (str, opcional): Texto inicial para dar contexto
            
        Retorna:
            str: Texto transcrito
        """
        pass
    
    def transcribe_from_recorder(self, recorder, output_wav: str = DEFAULT_OUTPUT_WAV, 
                                segment_length: int = SEGMENT_LENGTH) -> str:
        """
        Transcreve áudio do gravador, dividindo em segmentos se necessário.
        
        Parâmetros:
            recorder: Instância de AudioRecorder com os frames gravados
            output_wav (str): Caminho para salvar o arquivo WAV completo
            segment_length (int): Duração de cada segmento em segundos
            
        Retorna:
            str: Transcrição completa dividida por segmentos
        """
        import logging
        
        try:
            import librosa
            import numpy as np
            LIBROSA_AVAILABLE = True
        except ImportError:
            LIBROSA_AVAILABLE = False
            import wave
            import numpy as np
        
        logger = logging.getLogger("Transcriber")
        
        # Determina a largura da amostra (sample width) a partir do gravador
        sample_width = recorder.audio.get_sample_size(recorder.fmt)
        
        # Obtém uma cópia segura dos frames atuais
        frames = recorder.get_frames()
        
        if not frames:
            logger.warning("Não há frames para transcrever")
            return "Não há áudio para transcrever."
        
        # Salva todos os frames em um único arquivo WAV
        save_frames_to_wav(
            frames,
            output_wav,
            recorder.rate,
            recorder.channels,
            sample_width
        )
        
        # Prepara lista para armazenar textos de cada segmento
        segments_text = []
        
        # Tamanho de segmento curto para melhor precisão (6 segundos)
        smaller_segment_length = 6  # segundos
        
        # Define o overlap em segundos (50% do tamanho do segmento)
        overlap_seconds = smaller_segment_length // 2
        
        use_silence_detection = False
        
        # Tente usar librosa para detecção de silêncio se disponível
        if LIBROSA_AVAILABLE:
            try:
                # Carregar o áudio completo usando librosa
                y, sr = librosa.load(output_wav, sr=None)
                logger.info(f"Áudio carregado: {len(y)/sr:.2f} segundos a {sr}Hz")
                
                # Detecta períodos de silêncio para segmentação inteligente
                non_silent_intervals = librosa.effects.split(
                    y, 
                    top_db=36,       # Limiar de dB para considerar como silêncio (menos agressivo)
                    frame_length=512,  # Comprimento de janela para análise
                    hop_length=128     # Tamanho de salto entre janelas
                )
                
                # Se não foi possível detectar períodos sem silêncio, use a abordagem padrão
                if len(non_silent_intervals) == 0:
                    logger.info("Não foi possível detectar períodos de fala - usando segmentação fixa")
                    use_silence_detection = False
                else:
                    logger.info(f"Detectados {len(non_silent_intervals)} períodos de fala")
                    use_silence_detection = True
                    
            except Exception as e:
                logger.error(f"Erro ao processar áudio para detecção de silêncio: {e}")
                use_silence_detection = False
        else:
            logger.info("Librosa não disponível - usando segmentação fixa")
            
        # Abre o arquivo WAV completo para leitura
        with wave.open(output_wav, 'rb') as wf:
            # Obtém propriedades do arquivo WAV
            rate = wf.getframerate()       # Taxa de amostragem
            channels = wf.getnchannels()   # Número de canais
            sw = wf.getsampwidth()         # Largura da amostra
            n_frames = wf.getnframes()     # Número total de frames
            audio_duration = n_frames / rate  # Duração total em segundos
            
            logger.info(f"Arquivo WAV: {audio_duration:.2f}s, {rate}Hz, {channels} canais")
            
            # Contexto para manter continuidade entre segmentos
            context = None
            full_text_so_far = ""
            
            # Processa o áudio em segmentos
            segment_count = 0
            raw_segments = []
            
            # Se temos detecção de silêncio com librosa, use-a para segmentação inteligente
            if use_silence_detection:
                # O código de detecção de silêncio é mantido aqui para quando librosa estiver disponível
                # Processa cada intervalo não silencioso com overlaps
                processed_intervals = []
                
                # Agrupa intervalos próximos para evitar segmentos muito curtos
                merged_intervals = []
                current_interval = None
                
                max_gap = int(0.3 * sr)  # 300ms de silêncio máximo para considerar como mesmo segmento
                
                # Mescla intervalos que estão muito próximos
                for interval in non_silent_intervals:
                    if current_interval is None:
                        current_interval = interval
                    else:
                        # Se o intervalo atual está próximo do próximo, mescle-os
                        if interval[0] - current_interval[1] <= max_gap:
                            current_interval[1] = interval[1]
                        else:
                            merged_intervals.append(current_interval)
                            current_interval = interval
                
                # Adiciona o último intervalo
                if current_interval is not None:
                    merged_intervals.append(current_interval)
                
                logger.info(f"Após mesclagem: {len(merged_intervals)} segmentos de fala")
                
                # Divide intervalos longos em partes menores com sobreposição
                for interval in merged_intervals:
                    start_sample, end_sample = interval
                    interval_duration = (end_sample - start_sample) / sr
                    
                    # Se o intervalo for muito curto, expanda-o um pouco
                    if interval_duration < 1.0:  # Menos de 1 segundo
                        padding = int(0.5 * sr)  # Adiciona 500ms de cada lado
                        start_sample = max(0, start_sample - padding)
                        end_sample = min(len(y), end_sample + padding)
                        interval_duration = (end_sample - start_sample) / sr
                    
                    # Se o intervalo for mais longo que o tamanho de segmento, divida-o
                    if interval_duration > smaller_segment_length:
                        # Calcula quantos segmentos completos cabem neste intervalo
                        segment_samples = int(smaller_segment_length * sr)
                        overlap_samples = int(overlap_seconds * sr)
                        step = segment_samples - overlap_samples
                        
                        # Divide o intervalo em segmentos sobrepostos
                        for seg_start in range(start_sample, end_sample, step):
                            seg_end = min(seg_start + segment_samples, end_sample)
                            # Se o último segmento for muito curto, mescla com o anterior
                            if (end_sample - seg_start) < (0.5 * segment_samples) and len(processed_intervals) > 0:
                                processed_intervals[-1][1] = end_sample
                            else:
                                processed_intervals.append([seg_start, seg_end])
                            
                            # Evita segmentos muito curtos no final
                            if seg_end >= end_sample - (0.5 * segment_samples):
                                break
                    else:
                        # Intervalo curto, use-o diretamente
                        processed_intervals.append([start_sample, end_sample])
                
                logger.info(f"Segmentação final: {len(processed_intervals)} segmentos para processar")
                
                # Processa cada segmento
                for i, (start_sample, end_sample) in enumerate(processed_intervals):
                    segment_count += 1
                    
                    # Converte amostras para frames do arquivo WAV
                    start_frame = int((start_sample / sr) * rate)
                    num_frames = int(((end_sample - start_sample) / sr) * rate)
                    
                    # Garante que não ultrapassamos o limite do arquivo
                    start_frame = max(0, start_frame)
                    num_frames = min(num_frames, n_frames - start_frame)
                    
                    # Posiciona o ponteiro de leitura
                    wf.setpos(start_frame)
                    
                    # Lê o bloco de frames para este segmento
                    frames_chunk = wf.readframes(num_frames)
                    
                    # Cria um nome para o arquivo temporário deste segmento
                    seg_path = f"{TEMP_DIR}/{os.path.basename(output_wav).rstrip('.wav')}_seg{segment_count}.wav"
                    
                    # Salva este segmento como um arquivo WAV separado
                    save_frames_to_wav([frames_chunk], seg_path, rate, channels, sw)
                    
                    # Calcula a duração exata deste segmento
                    segment_duration = num_frames / rate
                    logger.info(f"Segmento {segment_count}: {segment_duration:.2f}s")
                    
                    # Pula segmentos extremamente curtos (menos de 0.5 segundos)
                    if segment_duration < 0.5:
                        logger.info(f"Segmento {segment_count} muito curto, pulando")
                        os.remove(seg_path)
                        continue
                    
                    # Usa contexto aprimorado para transcrição
                    seg_text = self.transcribe_file(seg_path, initial_prompt=context)
                    
                    # Se não obtivemos texto, tente aumentar a sensibilidade
                    if not seg_text.strip() and segment_duration > 1.0:
                        logger.info(f"Tentando novamente o segmento {segment_count} com configurações mais sensíveis")
                        os.remove(seg_path)
                        # Expande um pouco o segmento
                        expanded_start = max(0, start_frame - int(0.5 * rate))
                        expanded_frames = min(n_frames - expanded_start, num_frames + int(1.0 * rate))
                        
                        wf.setpos(expanded_start)
                        frames_chunk = wf.readframes(expanded_frames)
                        
                        save_frames_to_wav([frames_chunk], seg_path, rate, channels, sw)
                        
                        # Tenta transcrever novamente
                        seg_text = self.transcribe_file(seg_path, initial_prompt=context)
                    
                    # Armazena o texto bruto para pós-processamento
                    raw_segments.append(seg_text)
                    
                    # Atualiza o contexto para o próximo segmento
                    context = seg_text if seg_text.strip() else context
                    
                    # Atualiza o texto completo acumulado
                    if seg_text.strip():
                        if full_text_so_far:
                            full_text_so_far += " " + seg_text
                        else:
                            full_text_so_far = seg_text
                    
                    # Remove o arquivo temporário do segmento após uso
                    os.remove(seg_path)
            
            else:
                # Método tradicional com segmentos de tamanho fixo (funcionará sem librosa)
                # Calcula quantos frames correspondem a um segmento
                segment_frames = rate * smaller_segment_length
                
                # Calcula quantos frames correspondem ao overlap
                overlap_frames = rate * overlap_seconds
                
                # Passos menores para segmentos menores com maior sobreposição
                # Agora usando 75% de sobreposição para garantir continuidade
                effective_step = int(segment_frames * 0.25)  # 75% overlap
                
                for i in range(0, n_frames, effective_step):
                    segment_count += 1
                    
                    # Calcula a posição de início com overlap 
                    # (exceto para o primeiro segmento)
                    start_pos = max(0, i)
                    
                    # Calcula o número de frames a ler (com ajuste para não ultrapassar o fim do arquivo)
                    frames_to_read = min(segment_frames, n_frames - start_pos)
                    
                    # Se o último segmento for muito curto, anexe-o ao anterior
                    if frames_to_read < rate * 1.5 and segment_count > 1:  # Menos de 1.5 segundos
                        logger.info(f"Último segmento muito curto ({frames_to_read/rate:.2f}s), pulando")
                        continue
                    
                    # Posiciona o ponteiro de leitura no início do segmento atual
                    wf.setpos(start_pos)
                    
                    # Lê o bloco de frames para este segmento
                    frames_chunk = wf.readframes(frames_to_read)
                    
                    # Cria um nome para o arquivo temporário deste segmento
                    seg_path = f"{TEMP_DIR}/{os.path.basename(output_wav).rstrip('.wav')}_seg{segment_count}.wav"
                    
                    # Salva este segmento como um arquivo WAV separado
                    save_frames_to_wav([frames_chunk], seg_path, rate, channels, sw)
                    
                    # Tenta transcrever com contexto acumulado
                    seg_text = self.transcribe_file(seg_path, initial_prompt=context)
                    
                    # Se não obtivemos texto, tente com configurações mais sensíveis
                    # (adicionando um pouco mais de áudio antes e depois)
                    if not seg_text.strip() and frames_to_read > rate * 2:
                        extended_start = max(0, start_pos - int(rate * 0.5))
                        extended_length = min(n_frames - extended_start, 
                                             frames_to_read + int(rate * 1.0))
                        
                        # Remova o arquivo anterior e crie um novo estendido
                        os.remove(seg_path)
                        wf.setpos(extended_start)
                        extended_frames = wf.readframes(extended_length)
                        save_frames_to_wav([extended_frames], seg_path, rate, channels, sw)
                        
                        # Tente novamente a transcrição
                        seg_text = self.transcribe_file(seg_path, initial_prompt=context)
                    
                    # Armazena o texto bruto para pós-processamento
                    raw_segments.append(seg_text)
                    
                    # Atualiza o contexto para o próximo segmento
                    context = seg_text if seg_text.strip() else context
                    
                    # Atualiza o texto completo acumulado
                    if seg_text.strip():
                        if full_text_so_far:
                            full_text_so_far += " " + seg_text
                        else:
                            full_text_so_far = seg_text
                    
                    # Remove o arquivo temporário do segmento após uso
                    os.remove(seg_path)
            
            # Pós-processamento: Criar os segmentos finais com melhor fusão
            processed_segments = []
            
            for i, text in enumerate(raw_segments):
                if not text.strip():
                    continue  # Pula segmentos vazios
                    
                # Remove textos duplicados entre segmentos sobrepostos
                if i > 0:
                    # Tenta identificar sobreposições de frases entre segmentos
                    prev_text = raw_segments[i-1]
                    if not prev_text.strip():
                        continue
                        
                    # Tenta encontrar frases sobrepostas
                    words = prev_text.split()
                    
                    # Tenta diferentes tamanhos de sobreposição
                    for overlap_size in [8, 6, 4, 2]:
                        if len(words) >= overlap_size:
                            phrase = " ".join(words[-overlap_size:])
                            if text.startswith(phrase):
                                # Remove a parte sobreposta
                                text = text[len(phrase):].strip()
                                break
                
                # Adiciona o segmento processado
                processed_segments.append(f"Segment {i+1}:\n{text}")
        
        # Combina todas as transcrições dos segmentos com separação por linhas em branco
        if not processed_segments:
            return "Não foi possível transcrever o áudio."
            
        segmented_result = "\n\n".join(processed_segments)
        
        # Adiciona também a transcrição completa consolidada como alternativa
        return f"{segmented_result}\n\n===== TRANSCRIÇÃO COMPLETA =====\n{full_text_so_far}"

    def transcribe_microphone_realtime(self, recorder, chunk_duration: float = 2.0, callback = None):
        """
        Transcreve áudio do microfone em tempo real.
        
        Esta função configura um callback no gravador para processar chunks de áudio
        à medida que são capturados, sem esperar pela gravação completa.
        
        Parâmetros:
            recorder: Instância de AudioRecorder configurada para gravação
            chunk_duration (float): Duração em segundos de cada chunk a processar
            callback: Função que recebe o texto transcrito a cada atualização
                      Assinatura: callback(texto: str, é_final: bool)
            
        Nota:
            O recorder deve ser configurado para capturar áudio continuamente.
            Esta função não retorna diretamente o resultado, mas chama o callback
            fornecido com atualizações de texto a cada chunk processado.
        """
        # Buffer para manter o stream de áudio contínuo
        audio_stream = None
        
        # Timestamp do último processamento
        last_process_time = time.time()
        
        # Função para processar chunks de áudio
        def process_audio_chunk(frames, is_final=False):
            nonlocal audio_stream, last_process_time
            
            current_time = time.time()
            # Verifica se passamos tempo suficiente desde o último processamento
            time_elapsed = current_time - last_process_time
            
            # Processa apenas a cada chunk_duration segundos ou se for o chunk final
            if time_elapsed < chunk_duration and not is_final:
                return
                
            # Atualiza o timestamp
            last_process_time = current_time
            
            try:
                # Verifica se temos frames para processar
                if not frames or len(frames) == 0:
                    logging.warning("Nenhum frame de áudio para processar")
                    return
                
                # Abordagem simplificada para áudio mono
                try:
                    # 1. Criar um buffer de bytes único e converter para array numpy
                    audio_bytes = b''.join(frames)
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # 2. Verificar e ajustar o formato (estéreo para mono)
                    if recorder.channels == 2:
                        # Reorganizar para formato (n_samples, 2)
                        samples_per_frame = len(audio_data) // len(frames)
                        logging.info(f"Processando {len(frames)} frames, {samples_per_frame} amostras/frame")
                        
                        try:
                            # Reshape e média para mono
                            audio_data = audio_data.reshape(-1, 2)
                            audio_data = np.mean(audio_data, axis=1)
                            logging.info(f"Áudio estéreo convertido para mono: {audio_data.shape}")
                        except Exception as reshape_error:
                            logging.error(f"Erro ao converter estéreo para mono: {reshape_error}")
                            # Fallback: usar apenas um canal
                            audio_data = audio_data[::2]
                            logging.info(f"Fallback: usando apenas o canal esquerdo: {audio_data.shape}")
                    
                    # 3. Normalizar para o intervalo [-1, 1]
                    audio_float = audio_data.astype(np.float32) / 32768.0
                    
                    # 4. Garantir formato 1D consistente
                    if len(audio_float.shape) > 1:
                        audio_float = audio_float.flatten()
                    
                    # Acrescentar consistência extra: garantir tipo e formato corretos
                    audio_float = np.ascontiguousarray(audio_float, dtype=np.float32)
                    
                    # Log dos detalhes do áudio para diagnóstico
                    logging.info(f"Áudio preparado: shape={audio_float.shape}, "
                                f"min={np.min(audio_float):.2f}, max={np.max(audio_float):.2f}")
                    
                    # 5. Processar o stream com tratamento de erros específico
                    audio_stream, text = self.transcribe_stream(audio_float, audio_stream)
                    
                    # Chama o callback com o texto
                    if callback and text:
                        callback(text, is_final)
                        
                except RuntimeError as e:
                    # Captura específica para o erro de tensor mismatch
                    if "sizes of tensors must match" in str(e).lower():
                        logging.error(f"Erro de dimensão: {e} - reiniciando stream")
                        # Reiniciar o stream e processar apenas o chunk atual
                        audio_stream = None
                        audio_stream, text = self.transcribe_stream(audio_float, None)
                        
                        if callback and text:
                            callback(text, is_final)
                    else:
                        raise e
                
            except Exception as e:
                logging.error(f"Erro ao processar chunk de áudio: {e}")
                if callback:
                    callback(f"[ERRO: {str(e)}]", is_final)
        
        # Teste do callback para garantir que a UI será atualizada
        if callback:
            callback("Transcrição iniciada. Aguardando áudio...", False)
        
        # Atribui a função de callback e a duração do chunk ao recorder
        recorder.set_realtime_transcription(True, chunk_duration, callback)
        
        # O callback será chamado pelo próprio recorder durante a gravação
        # quando tiver chunks completos para transcrever


def save_frames_to_wav(frames, path, rate, channels, sample_width):
    """
    Salva frames de áudio brutos em um arquivo WAV.
    
    Parâmetros:
        frames (list): Lista de buffers de áudio (bytes)
        path (str): Caminho onde o arquivo WAV será salvo
        rate (int): Taxa de amostragem do áudio (amostras por segundo)
        channels (int): Número de canais (1=mono, 2=estéreo)
        sample_width (int): Largura da amostra em bytes (2 para 16 bits)
    """
    # Cria o diretório de destino se não existir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Abre o arquivo WAV para escrita em modo binário
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)       # Define número de canais (mono/estéreo)
        wf.setsampwidth(sample_width)   # Define tamanho das amostras em bytes
        wf.setframerate(rate)           # Define taxa de amostragem (Hz)
        wf.writeframes(b"".join(frames)) # Escreve todos os frames concatenados


# ====== Funções e objetos anteriormente em audio_transcriber.py ======

# Funções de conveniência para manter compatibilidade com código existente 
# que usa a API anterior

def get_default_transcriber():
    """
    Obtém a instância do transcritor padrão.
    Importa o WhisperTranscriber aqui para evitar importação circular.
    """
    from whisper_transcriber import WhisperTranscriber
    global default_transcriber
    
    if 'default_transcriber' not in globals() or default_transcriber is None:
        default_transcriber = WhisperTranscriber()
        
    return default_transcriber

def transcribe_audio(file_path=None, initial_prompt=None, frames=None, sample_rate=None) -> str:
    """
    Transcreve um arquivo de áudio ou dados de áudio em frames usando o transcritor padrão.
    Função auxiliar para manter compatibilidade com código antigo.
    
    Parâmetros:
        file_path (str, opcional): Caminho para o arquivo de áudio
        initial_prompt (str, opcional): Texto inicial para dar contexto
        frames (np.ndarray, opcional): Frames de áudio como array numpy
        sample_rate (int, opcional): Taxa de amostragem dos frames
        
    Retorna:
        str: Texto transcrito
    """
    transcriber = get_default_transcriber()
    
    # Se forneceu frames de áudio diretamente, processa-os
    if frames is not None and sample_rate is not None:
        return transcriber.transcribe(frames, sample_rate)
    
    # Caso contrário, transcreve um arquivo
    elif file_path is not None:
        return transcriber.transcribe_file(file_path, initial_prompt)
    
    else:
        raise ValueError("Deve fornecer file_path OU (frames E sample_rate)")

def transcribe_from_recorder(recorder, output_wav=None, segment_length=None):
    """
    Transcreve áudio do gravador usando o transcritor padrão.
    Função auxiliar para manter compatibilidade com código antigo.
    """
    transcriber = get_default_transcriber()
    
    kwargs = {}
    if output_wav is not None:
        kwargs['output_wav'] = output_wav
    if segment_length is not None:
        kwargs['segment_length'] = segment_length
        
    return transcriber.transcribe_from_recorder(recorder, **kwargs)

def transcribe_microphone_realtime(recorder, chunk_duration=2.0, callback=None):
    """
    Configura o transcritor para processar o áudio do microfone em tempo real.
    
    Args:
        recorder: Instância do AudioRecorder
        chunk_duration: Duração de cada chunk de áudio a ser transcrito (em segundos)
        callback: Função a ser chamada quando um texto é transcrito
               Assinatura: callback(texto, is_final)
    
    Returns:
        None
    """
    logger.info(f"Iniciando transcrição em tempo real com chunks de {chunk_duration}s")
    
    # Teste do callback para garantir que a UI será atualizada
    if callback:
        callback("Transcrição iniciada. Aguardando áudio...", False)
    
    # Atribui a função de callback e a duração do chunk ao recorder
    recorder.set_realtime_transcription(True, chunk_duration, callback)
    
    # O callback será chamado pelo próprio recorder durante a gravação
    # quando tiver chunks completos para transcrever 
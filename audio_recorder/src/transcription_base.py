# transcription_base.py
# Módulo base para diferentes tipos de transcritores de áudio para texto
# Define interfaces, funções comuns e funções de conveniência

import os
import wave
import abc
import sys

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    SEGMENT_LENGTH,    # Duração de cada segmento para processamento (segundos)
    TEMP_DIR,          # Diretório para arquivos temporários
    DEFAULT_OUTPUT_WAV # Caminho padrão para o arquivo WAV de saída
)


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
        # Determina a largura da amostra (sample width) a partir do gravador
        sample_width = recorder.audio.get_sample_size(recorder.fmt)
        
        # Obtém uma cópia segura dos frames atuais
        frames = recorder.get_frames()
        
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
        
        # Abre o arquivo WAV completo para leitura
        with wave.open(output_wav, 'rb') as wf:
            # Obtém propriedades do arquivo WAV
            rate = wf.getframerate()       # Taxa de amostragem
            channels = wf.getnchannels()   # Número de canais
            sw = wf.getsampwidth()         # Largura da amostra
            n_frames = wf.getnframes()     # Número total de frames
            
            # Calcula quantos frames correspondem a um segmento
            segment_frames = rate * segment_length
            
            # Contexto para manter continuidade entre segmentos
            context = None
            
            # Processa o áudio em segmentos
            for i in range(0, n_frames, segment_frames):
                # Posiciona o ponteiro de leitura no início do segmento atual
                wf.setpos(i)
                
                # Lê o bloco de frames para este segmento
                frames_chunk = wf.readframes(segment_frames)
                
                # Cria um nome para o arquivo temporário deste segmento
                seg_path = f"{TEMP_DIR}/{os.path.basename(output_wav).rstrip('.wav')}_seg{i//segment_frames}.wav"
                
                # Salva este segmento como um arquivo WAV separado
                save_frames_to_wav([frames_chunk], seg_path, rate, channels, sw)
                
                # Transcreve o segmento em texto com contexto do segmento anterior
                seg_text = self.transcribe_file(seg_path, initial_prompt=context)
                
                # Adiciona o texto com um cabeçalho de segmento
                segments_text.append(f"Segment {i//segment_frames + 1}:\n{seg_text}")
                
                # Atualiza o contexto para o próximo segmento
                context = seg_text
                
                # Remove o arquivo temporário do segmento após uso
                os.remove(seg_path)
                
        # Combina todas as transcrições dos segmentos com separação por linhas em branco
        return "\n\n".join(segments_text)


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

# Importação no final para evitar importação circular
from whisper_transcriber import WhisperTranscriber

# Cria uma instância do transcritor padrão (Whisper) para uso
default_transcriber = WhisperTranscriber()

# Funções de conveniência para manter compatibilidade com código existente 
# que usa a API anterior

def transcribe_audio(file_path: str, initial_prompt: str = None) -> str:
    """
    Transcreve um arquivo de áudio usando o transcritor padrão.
    Função auxiliar para manter compatibilidade com código antigo.
    """
    return default_transcriber.transcribe_file(file_path, initial_prompt)

def transcribe_from_recorder(recorder, output_wav=None, segment_length=None):
    """
    Transcreve áudio do gravador usando o transcritor padrão.
    Função auxiliar para manter compatibilidade com código antigo.
    """
    kwargs = {}
    if output_wav is not None:
        kwargs['output_wav'] = output_wav
    if segment_length is not None:
        kwargs['segment_length'] = segment_length
        
    return default_transcriber.transcribe_from_recorder(recorder, **kwargs) 
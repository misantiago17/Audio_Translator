# whisper_transcriber.py
# Implementação específica para transcrição usando o modelo Whisper da OpenAI

import whisper
import sys
import os

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import WHISPER_MODEL

# Importa a classe base
from transcription_base import AudioTranscriber

class WhisperTranscriber(AudioTranscriber):
    """
    Transcritor de áudio usando o modelo Whisper da OpenAI.
    
    Este transcritor usa o modelo Whisper localmente para converter
    áudio em texto, otimizado para arquivos WAV em 16kHz mono.
    """
    
    def __init__(self, model_name=WHISPER_MODEL):
        """
        Inicializa o transcritor Whisper com o modelo especificado.
        
        Parâmetros:
            model_name (str): Nome do modelo Whisper a carregar
                             (tiny, base, small, medium, large)
        """
        # Carrega o modelo Whisper localmente na CPU
        # Isso pode levar alguns segundos
        try:
            # Tenta carregar com configuração float32 para evitar avisos de fallback para FP16
            self.model = whisper.load_model(model_name, device="cpu", compute_type="float32")
        except TypeError:
            # Versões antigas do Whisper não suportam o parâmetro compute_type
            self.model = whisper.load_model(model_name, device="cpu")
    
    def transcribe_file(self, file_path: str, initial_prompt: str = None) -> str:
        """
        Transcreve um arquivo de áudio WAV usando o modelo Whisper.
        
        Parâmetros:
            file_path (str): Caminho para o arquivo WAV a ser transcrito
            initial_prompt (str, opcional): Texto inicial para dar contexto
            
        Retorna:
            str: Texto transcrito do áudio
        """
        # Carrega o áudio usando a função nativa do Whisper que cuida da normalização
        audio_array = whisper.load_audio(file_path)
        
        # Transcreve o áudio com o modelo Whisper
        if initial_prompt:
            # Usa o texto anterior como prompt para manter continuidade
            result = self.model.transcribe(
                audio_array, 
                initial_prompt=initial_prompt, 
                condition_on_previous_text=True
            )
        else:
            # Transcrição sem contexto prévio
            result = self.model.transcribe(audio_array)
        
        # Limpa e retorna o texto, removendo espaços extras
        text = result.get("text", "").strip()
        
        # Remove reticências (...) no final, que podem atrapalhar a concatenação
        if text.endswith("..."):
            text = text[:-3].rstrip()
        
        return text 
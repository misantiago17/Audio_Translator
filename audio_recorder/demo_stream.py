#!/usr/bin/env python3
import time
import logging
import argparse
import sys
from threading import Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_recorder.src.whisper_transcriber import WhisperTranscriber
from audio_recorder.src.stream_recorder import StreamRecorder
import torch

def main():
    parser = argparse.ArgumentParser(description="Demo de transcrição de áudio em tempo real")
    parser.add_argument("--model", type=str, default="tiny", help="Modelo Whisper a ser usado (tiny, base, small, medium)")
    parser.add_argument("--language", type=str, default="pt", help="Código do idioma para transcrição")
    parser.add_argument("--device", type=int, default=None, help="Índice do dispositivo de áudio (microfone)")
    parser.add_argument("--duration", type=int, default=30, help="Duração da gravação em segundos")
    args = parser.parse_args()

    # Inicializa o transcritor
    logging.info(f"Inicializando modelo Whisper {args.model}...")
    
    # Verifica se CUDA está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Usando dispositivo: {device}")
    
    try:
        # Cria o transcritor com as opções corretas
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=device
        )
        
        # Define o idioma após a inicialização
        transcriber.set_language(args.language)
        
        # Inicializa o gravador de stream
        recorder = StreamRecorder(
            transcriber=transcriber,
            device_index=args.device,
            callback_interval=0.5  # Atualiza a cada 0.5 segundos
        )
        
        # Adiciona callback para exibir a transcrição em tempo real
        def on_transcription(text):
            print(f"\r\033[KTranscrição: {text}", end="", flush=True)
        
        recorder.add_transcription_callback(on_transcription)
        
        # Inicia a gravação
        print(f"Gravando por {args.duration} segundos... (Ctrl+C para interromper)")
        print(f"Usando modelo: {args.model}, Idioma: {args.language}, Dispositivo: {device}")
        recorder.start()
        
        # Aguarda duração especificada
        time.sleep(args.duration)
        
        # Para a gravação
        recorder.stop()
        print("\nGravação finalizada.")
        
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        if 'recorder' in locals():
            recorder.stop()
    except Exception as e:
        logging.error(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Garante que todos os recursos sejam liberados
        if 'recorder' in locals():
            del recorder
        if 'transcriber' in locals():
            del transcriber

if __name__ == "__main__":
    main() 
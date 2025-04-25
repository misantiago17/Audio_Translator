"""
Interface Gráfica para o Gravador e Transcritor de Áudio
--------------------------------------------------------
Este módulo implementa uma interface gráfica simples usando tkinter
para capturar áudio e transcrevê-lo em texto usando o modelo Whisper.
Suporta transcrição em tempo real e em lote (após a gravação).
"""

# Importações de bibliotecas padrão
import tkinter as tk            # Framework de GUI
from tkinter import scrolledtext # Widget para exibir e rolar texto
import threading                # Para operações em segundo plano
import time, os                 # Para operações de temporização e sistema de arquivos
import sys                      # Para manipulação de caminhos de sistema
import logging                  # Para logs

# Adiciona o diretório raiz ao caminho de busca para importar constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import (
    WINDOW_SECONDS,  # Define quanto tempo de áudio será mantido no buffer
    SEGMENT_LENGTH,  # Tamanho de cada segmento de áudio para processamento
    HOP_LENGTH,      # Intervalo entre os inícios de segmentos consecutivos 
    TEMP_DIR         # Diretório para arquivos temporários
)

# Importações de módulos do projeto
from audio_recorder import AudioRecorder  # Classe para captura de áudio
# Importamos da transcription_base que agora contém as funções anteriormente em audio_transcriber
from transcription_base import save_frames_to_wav, transcribe_audio, get_default_transcriber, transcribe_microphone_realtime

class AudioRecorderGUI:
    """
    Classe principal para a interface gráfica do gravador e transcritor.
    Gerencia a interação do usuário, a gravação de áudio e a transcrição.
    """
    def __init__(self, root):
        """
        Inicializa a GUI e configura todos os componentes.
        
        Args:
            root: Janela principal do Tkinter
        """
        self.root = root
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("AudioRecorderGUI")
        
        # Criar o gravador com a janela de buffer configurada (mantém WINDOW_SECONDS de áudio)
        self.rec = AudioRecorder(window_seconds=WINDOW_SECONDS)
        # Referência ao transcritor - pode ser alterada em tempo de execução se necessário
        self.transcriber = get_default_transcriber()
        
        # Variável para armazenar o texto acumulado durante o streaming
        self.accumulated_streaming_text = ""
        
        # Configuração da janela principal
        root.title("Gravador de Áudio do Sistema")
        root.geometry("300x320")  # Tamanho inicial da janela
        
        # Variável de controle para o status da aplicação
        self.status = tk.StringVar(value="Pronto")
        
        # Flag para indicar se estamos usando o modo de streaming
        self.streaming_mode = False
        
        # Criação dos widgets
        # Label para mostrar o status atual
        tk.Label(root, textvariable=self.status).pack(pady=10)
        
        # Botão de gravação (ativa/desativa a captura de áudio)
        self.record_btn = tk.Button(root, text="Gravar", fg="red", command=self.toggle)
        self.record_btn.pack(fill=tk.X, padx=20)
        
        # Botão de transcrição em streaming (novo modo)
        self.stream_btn = tk.Button(root, text="Gravar com Streaming", fg="purple", 
                                   command=self.toggle_streaming)
        self.stream_btn.pack(fill=tk.X, padx=20, pady=5)
        
        # Botão de transcrição (processa todo o áudio capturado)
        self.trans_btn = tk.Button(root, text="Transcrever", fg="blue", 
                                  command=self.transcribe, state=tk.DISABLED)
        self.trans_btn.pack(fill=tk.X, padx=20)
        
        # Botão para limpar o texto transcrito
        tk.Button(root, text="Limpar", fg="black", command=self.clear_text).pack(fill=tk.X, padx=20, pady=5)
        
        # Painel de texto rolável para mostrar a transcrição
        self.txt = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=10)
        self.txt.config(state=tk.NORMAL)
        self.txt.insert(tk.END, "Aplicativo pronto para capturar áudio do sistema.")
        self.txt.config(state=tk.DISABLED)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configura handler para o fechamento da janela
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Garante que o diretório temporário existe
        os.makedirs(TEMP_DIR, exist_ok=True)

    def toggle(self):
        """
        Alterna entre iniciar e parar a gravação de áudio.
        Quando inicia a gravação, também inicia a transcrição em tempo real.
        """
        if not self.rec.get_recording_status():
            # Inicia a gravação
            self.status.set("Gravando...")
            
            # Limpa transcrição anterior
            self.txt.config(state=tk.NORMAL)
            self.txt.delete("1.0", tk.END)
            self.txt.config(state=tk.DISABLED)
            
            # Inicia a gravação e a transcrição em tempo real em threads separadas
            # daemon=True faz as threads terminarem quando o programa principal termina
            threading.Thread(target=self.rec.start_recording, daemon=True).start()
            threading.Thread(target=self._live_transcribe, daemon=True).start()
            
            # Desabilita o botão de streaming enquanto está gravando
            self.stream_btn.config(state=tk.DISABLED)
        else:
            # Para a gravação
            self.rec.stop_recording()
            self.status.set("Pronto")
            # Habilita o botão de transcrição completa
            self.trans_btn.config(state=tk.NORMAL)
            # Habilita o botão de streaming novamente
            self.stream_btn.config(state=tk.NORMAL)
            
    def toggle_streaming(self):
        """
        Alterna entre iniciar e parar a gravação com transcrição em streaming.
        Utiliza a nova funcionalidade de processamento em tempo real.
        """
        if not self.rec.get_recording_status():
            # Inicia a gravação no modo streaming
            self.streaming_mode = True
            self.status.set("Gravando (Streaming)...")
            
            # Limpa transcrição anterior
            self.txt.config(state=tk.NORMAL)
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, "Iniciando transcrição...\nO texto aparecerá aqui em breve.\n\nUsando modelo Whisper 'base' para economizar recursos.")
            self.txt.config(state=tk.DISABLED)
            
            # Reinicia o texto acumulado
            self.accumulated_streaming_text = ""
            
            # Configura o callback para receber as transcrições em tempo real
            transcribe_microphone_realtime(
                self.rec, 
                chunk_duration=3.0,  # Aumentado para reduzir processamento
                callback=self._streaming_transcription_callback
            )
            
            # Inicia a gravação
            threading.Thread(target=self.rec.start_recording, daemon=True).start()
            
            # Altera o botão para "Parar Streaming"
            self.stream_btn.config(text="Parar Streaming")
            # Desabilita o botão de gravação normal enquanto estiver em streaming
            self.record_btn.config(state=tk.DISABLED)
        else:
            # Para a gravação
            self.rec.stop_recording()
            self.streaming_mode = False
            self.status.set("Pronto")
            
            # Restaura o texto original do botão
            self.stream_btn.config(text="Gravar com Streaming")
            # Habilita o botão de gravação normal novamente
            self.record_btn.config(state=tk.NORMAL)
            # Habilita o botão de transcrição completa
            self.trans_btn.config(state=tk.NORMAL)
            
    def _streaming_transcription_callback(self, text, is_final):
        """
        Callback para receber texto transcrito em tempo real do modo streaming.
        
        Args:
            text (str): Texto transcrito
            is_final (bool): Se é o texto final da gravação
        """
        if not text:
            return
        
        # Adiciona mensagem de debug (mais econômica)
        if len(text) > 50:
            self.logger.info(f"Recebido texto de {len(text)} caracteres")
        
        # Evitar atualizar a UI com muita frequência (economizar recursos)
        # Armazenamos o texto recebido e atualizamos a interface de forma controlada
        # Usa after_idle para otimizar ainda mais
        self.accumulated_streaming_text = text
        self.root.after_idle(lambda: self._update_streaming_text(text, is_final))
        
    def _update_streaming_text(self, text, is_final):
        """
        Atualiza a interface com o texto transcrito em streaming.
        
        Args:
            text (str): Texto transcrito
            is_final (bool): Se é o texto final da gravação
        """
        # Garante que temos um texto para mostrar
        if not text:
            return
            
        # Habilita a edição do texto
        self.txt.config(state=tk.NORMAL)
        
        # Se é o texto final, adiciona uma marca
        if is_final:
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, text + "\n\n--- Transcrição Completa ---")
        else:
            # Atualiza o texto acumulado
            self.accumulated_streaming_text = text
            
            # Mostra o texto na interface
            self.txt.delete("1.0", tk.END)
            self.txt.insert(tk.END, self.accumulated_streaming_text)
        
        # Rola para o final do texto
        self.txt.see(tk.END)
        self.txt.config(state=tk.DISABLED)
        
        # Forçar atualização imediata da janela
        self.root.update_idletasks()

    def transcribe(self):
        """
        Inicia a transcrição completa do áudio gravado em uma thread separada.
        Desabilita o botão de transcrição durante o processamento.
        """
        self.status.set("Transcrevendo...")
        self.trans_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._transcribe_thread, daemon=True).start()

    def _transcribe_thread(self):
        """
        Thread de trabalho para transcrição completa.
        Usa o transcritor para processar todo o áudio gravado.
        """
        try:
            # Usa o transcritor para processar todo o áudio gravado
            text = self.transcriber.transcribe_from_recorder(self.rec)
        except Exception as e:
            # Captura e exibe qualquer erro que ocorra durante a transcrição
            text = f"Erro: {e}"
            
        # Usa o loop de eventos do Tkinter para atualizar a UI de forma segura
        self.root.after(0, lambda: self._show_transcription(text))

    def _show_transcription(self, text):
        """
        Exibe o texto transcrito no painel principal.
        
        Args:
            text (str): Texto transcrito a ser exibido
        """
        self.status.set("Pronto")
        
        # Atualiza o painel de texto
        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, text)
        self.txt.config(state=tk.DISABLED)

    def _append_segment_text(self, idx, text):
        """
        Adiciona texto de um segmento transcrito ao painel de forma contínua.
        
        Args:
            idx (int): Índice do segmento
            text (str): Texto transcrito do segmento
        """
        self.txt.config(state=tk.NORMAL)
        # Adiciona texto do segmento sem rótulos para um fluxo contínuo
        self.txt.insert(tk.END, text + " ")
        self.txt.config(state=tk.DISABLED)

    def _live_transcribe(self):
        """
        Realiza transcrição em tempo real com janelas sobrepostas.
        
        Esta função implementa a técnica de "janelas deslizantes sobrepostas":
        1. Processa segmentos de áudio de tamanho SEGMENT_LENGTH
        2. Move a janela de processamento a cada HOP_LENGTH segundos
        3. Usa o contexto anterior para melhorar a continuidade da transcrição
        """
        # Configurações de segmentação
        segment_length = SEGMENT_LENGTH  # segundos
        hop_length = HOP_LENGTH          # segundos
        
        # Obtém parâmetros de áudio do gravador
        rate = self.rec.rate             # Taxa de amostragem
        channels = self.rec.channels     # Número de canais
        fmt = self.rec.fmt               # Formato de áudio
        sample_width = self.rec.audio.get_sample_size(fmt)  # Tamanho da amostra
        
        # Calcula quantos buffers correspondem a um segmento e um hop
        buffers_per_segment = int(rate * segment_length / self.rec.chunk)
        buffers_per_hop = int(rate * hop_length / self.rec.chunk)
        
        # Inicializa contexto vazio (será usado para continuidade entre segmentos)
        context = ""
        
        # Contador para identificar os segmentos temporários
        segment_counter = 0
        
        # Inicializa o texto completo da transcrição
        full_text = ""
        
        # Limpa o painel de texto para a nova transcrição
        self.root.after(0, self.clear_text)
        
        # Loop principal de transcrição em tempo real
        while self.rec.get_recording_status():
            # Obtém cópia segura dos frames atuais
            frames = self.rec.get_frames()
            total_buffers = len(frames)
            
            # Verifica se temos áudio suficiente para processar
            if total_buffers >= buffers_per_segment:
                # Extrai os últimos N buffers para formar o segmento atual
                seg_buffers = frames[-buffers_per_segment:]
                
                # Cria um arquivo WAV temporário para este segmento
                seg_path = f"{TEMP_DIR}/live_seg_{segment_counter}.wav"
                save_frames_to_wav(seg_buffers, seg_path, rate, channels, sample_width)
                
                # Transcreve o segmento, usando o contexto anterior para continuidade
                segment_text = self.transcriber.transcribe_file(seg_path, initial_prompt=context)
                
                # Remove o arquivo temporário após uso
                os.remove(seg_path)
                
                # Determina o novo texto a adicionar baseado no contexto anterior
                new_text = self._extract_new_content(context, segment_text)
                
                # Atualiza o contexto para o próximo segmento
                context = segment_text
                
                # Atualiza o texto completo
                if new_text:
                    full_text += " " + new_text if full_text else new_text
                    
                    # Atualiza a UI com o texto incrementalmente
                    self.root.after(0, lambda t=new_text: self._append_text(t + " "))
                
                # Incrementa o contador de segmentos
                segment_counter += 1
                
                # Espera pelo próximo hop (metade do tempo para permitir sobreposição)
                time.sleep(hop_length / 2)
            else:
                # Se não temos áudio suficiente, aguarda um pouco
                time.sleep(0.2)
                
        # Processa qualquer áudio restante após parar a gravação
        frames = self.rec.get_frames()
        if len(frames) > 0:
            # Salva o áudio restante em um arquivo temporário
            seg_path = f"{TEMP_DIR}/live_seg_last.wav"
            save_frames_to_wav(frames, seg_path, rate, channels, sample_width)
            
            # Transcreve o último segmento
            last_text = self.transcriber.transcribe_file(seg_path, initial_prompt=context)
            
            # Remove o arquivo temporário
            os.remove(seg_path)
            
            # Extrai apenas o novo conteúdo
            new_text = self._extract_new_content(context, last_text)
            
            # Atualiza o texto final se houver conteúdo novo
            if new_text:
                full_text += " " + new_text if full_text else new_text
                # Adiciona o último trecho ao painel
                self.root.after(0, lambda t=new_text: self._append_text(t))
    
    def _extract_new_content(self, previous_text, current_text):
        """
        Extrai apenas o conteúdo novo entre duas transcrições consecutivas.
        
        Args:
            previous_text (str): Texto transcrito anteriormente
            current_text (str): Texto da transcrição atual
            
        Returns:
            str: Apenas o texto novo que não estava na transcrição anterior
        """
        # Se não há texto anterior, todo o texto atual é novo
        if not previous_text:
            return current_text
            
        # Se os textos são idênticos, não há conteúdo novo
        if previous_text == current_text:
            return ""
            
        # Tenta encontrar onde o texto anterior termina no texto atual
        # para extrair apenas o conteúdo novo
        words_previous = previous_text.split()
        words_current = current_text.split()
        
        # Se o texto atual é menor, algo estranho aconteceu - retorna ele completo
        if len(words_current) < len(words_previous):
            return current_text
            
        # Encontra o índice onde o texto novo começa
        # Permite uma margem de erro para pequenas alterações no texto anterior
        overlap_threshold = min(len(words_previous), 5)  # Usa até 5 palavras para verificar sobreposição
        
        for i in range(len(words_current) - overlap_threshold + 1):
            # Verifica se as últimas palavras do texto anterior correspondem 
            # a um trecho do texto atual
            match = True
            for j in range(min(overlap_threshold, len(words_previous))):
                if i+j >= len(words_current) or j >= len(words_previous) or words_current[i+j] != words_previous[-min(overlap_threshold, len(words_previous))+j]:
                    match = False
                    break
            
            if match:
                # Encontrou onde o texto anterior termina no atual
                # Retorna apenas o texto novo que vem depois
                return " ".join(words_current[i+overlap_threshold:])
        
        # Se não conseguiu encontrar uma sobreposição clara,
        # retorna a segunda metade do texto atual como aproximação
        half_point = len(words_current) // 2
        return " ".join(words_current[half_point:])

    def _append_text(self, text):
        """
        Adiciona texto ao painel sem limpar o conteúdo anterior.
        
        Args:
            text (str): Texto a ser adicionado ao painel
        """
        if not text.strip():
            return
            
        self.txt.config(state=tk.NORMAL)
        self.txt.insert(tk.END, text)
        self.txt.see(tk.END)  # Rola para mostrar o texto mais recente
        self.txt.config(state=tk.DISABLED)

    def on_close(self):
        """
        Manipula o evento de fechamento da janela.
        Para a gravação, libera recursos e fecha a aplicação.
        """
        # Para a gravação se estiver em andamento
        if self.rec.get_recording_status():
            self.rec.stop_recording()
            
        # Libera recursos do PyAudio
        self.rec.cleanup()
        
        # Destrói a janela
        self.root.destroy()

    def clear_text(self):
        """
        Limpa o painel de texto de transcrição.
        """
        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.config(state=tk.DISABLED)

# Ponto de entrada quando o script é executado diretamente
if __name__ == '__main__':
    # Cria a janela principal do Tkinter
    root = tk.Tk()
    # Inicializa nossa aplicação com a janela
    AudioRecorderGUI(root)
    # Inicia o loop principal de eventos da interface
    root.mainloop() 
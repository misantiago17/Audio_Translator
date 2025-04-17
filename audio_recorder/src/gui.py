import tkinter as tk
from tkinter import scrolledtext
import threading
from audio_recorder import AudioRecorder

class AudioRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.rec = AudioRecorder()
        root.title("Gravador")
        root.geometry("300x250")
        self.status = tk.StringVar(value="Pronto")
        tk.Label(root, textvariable=self.status).pack(pady=10)
        tk.Button(root, text="Gravar", fg="red", command=self.toggle).pack(fill=tk.X, padx=20)
        self.play_btn = tk.Button(root, text="Reproduzir", fg="green", command=self.play, state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, padx=20, pady=5)
        self.trans_btn = tk.Button(root, text="Transcrever", fg="blue", command=self.transcribe, state=tk.DISABLED)
        self.trans_btn.pack(fill=tk.X, padx=20)
        # Panel for showing transcription in the main window
        self.txt = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=10)
        self.txt.config(state=tk.DISABLED)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle(self):
        if not self.rec.get_recording_status():
            self.status.set("Gravando...")
            threading.Thread(target=self.rec.start_recording, daemon=True).start()
        else:
            self.rec.stop_recording()
            self.status.set("Pronto")
            self.play_btn.config(state=tk.NORMAL)
            self.trans_btn.config(state=tk.NORMAL)

    def play(self):
        self.status.set("Reproduzindo...")
        self.rec.play_recording()
        self.status.set("Pronto")

    def transcribe(self):
        """Trigger transcription in a background thread and show result."""
        self.status.set("Transcrevendo...")
        self.play_btn.config(state=tk.DISABLED)
        self.trans_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._transcribe_thread, daemon=True).start()

    def _transcribe_thread(self):
        try:
            from audio_transcriber import transcribe_from_recorder
            text = transcribe_from_recorder(self.rec)
        except Exception as e:
            text = f"Erro: {e}"
        self.root.after(0, lambda: self._show_transcription(text))

    def _show_transcription(self, text):
        """Display the transcription in the main window panel."""
        self.status.set("Pronto")
        # Insert text into the ScrolledText panel
        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, text)
        self.txt.config(state=tk.DISABLED)

    def on_close(self):
        if self.rec.get_recording_status():
            self.rec.stop_recording()
        self.rec.cleanup()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    AudioRecorderGUI(root)
    root.mainloop() 
import tkinter as tk
import threading
from audio_recorder import AudioRecorder

class AudioRecorderGUI:
    def __init__(self, root):
        self.rec = AudioRecorder()
        root.title("Gravador")
        root.geometry("300x200")
        self.status = tk.StringVar(value="Pronto")
        tk.Label(root, textvariable=self.status).pack(pady=10)
        tk.Button(root, text="Gravar", fg="red", command=self.toggle).pack(fill=tk.X, padx=20)
        self.play_btn = tk.Button(root, text="Reproduzir", fg="green", command=self.play, state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, padx=20, pady=5)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle(self):
        if not self.rec.get_recording_status():
            self.status.set("Gravando...")
            threading.Thread(target=self.rec.start_recording, daemon=True).start()
        else:
            self.rec.stop_recording()
            self.status.set("Pronto")
            self.play_btn.config(state=tk.NORMAL)

    def play(self):
        self.status.set("Reproduzindo...")
        self.rec.play_recording()
        self.status.set("Pronto")

    def on_close(self):
        if self.rec.get_recording_status():
            self.rec.stop_recording()
        self.rec.cleanup()
        root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    AudioRecorderGUI(root)
    root.mainloop() 
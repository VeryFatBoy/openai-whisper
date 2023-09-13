import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import pyaudio
import pyttsx3
import threading
import tkinter as tk
import queue
import wave
import whisper
from langchain import OpenAI, SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import scrolledtext

AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_RATE = 16000
CHUNK = 1024

# Get OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

s2_password = "<password>"
s2_host = "<host>"
s2_db = "timeseries_db"
db = SQLDatabase.from_uri(f"mysql+pymysql://admin:{s2_password}@{s2_host}:3306/{s2_db}")

llm = OpenAI(temperature = 0, verbose = False)
toolkit = SQLDatabaseToolkit(db = db, llm = llm)
agent_executor = create_sql_agent(llm = OpenAI(temperature = 0), toolkit = toolkit, verbose = False)

model = whisper.load_model("base.en")

# GUI class
class AudioRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")

        self.start_button = tk.Button(root, text = "Start Recording", command = self.start_recording)
        self.start_button.pack(pady = 10)

        self.stop_button = tk.Button(root, text = "Stop Recording", command = self.stop_recording, state = tk.DISABLED)
        self.stop_button.pack(pady = 5)

        self.exit_button = tk.Button(root, text = "Exit", command = self.exit_program)
        self.exit_button.pack(pady = 5)

        self.transcription_box = scrolledtext.ScrolledText(root, height = 5, width = 60)
        self.transcription_box.pack(padx = 10, pady = 10)

        self.recording_timer = None
        self.audio_filename = "audio_recording.wav"

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side = tk.TOP, fill = tk.BOTH, expand = 1)
        self.audio_array = np.array([])

        self.update_waveform = False

    def update_waveform_plot(self):
        while self.update_waveform and not self.stop_event.is_set():
            data = self.audio_queue.queue[-1] if not self.audio_queue.empty() else np.zeros(1024)
            self.audio_array = np.frombuffer(data, dtype = np.int16)
            self.ax.clear()
            self.ax.plot(self.audio_array, color = "r")
            self.ax.set_title("Real-Time Audio Waveform")
            self.ax.set_xlabel("Time (samples)")
            self.ax.set_ylabel("Amplitude")
            self.ax.set_ylim(-128, 128)
            self.ax.set_xlim(0, len(self.audio_array))
            self.canvas.draw()
            self.root.update()

    def speak_audio(self, text):
        engine = pyttsx3.init()
        engine.setProperty("voice", "english_us")
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def start_recording(self):
        self.transcription_box.delete(1.0, tk.END)
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()

        self.record_thread = threading.Thread(target = self.record_audio)
        self.record_thread.start()

        self.recording_timer = self.root.after(20000, self.stop_recording)

        self.update_waveform = True
        self.update_waveform_plot_thread = threading.Thread(target = self.update_waveform_plot)
        self.update_waveform_plot_thread.start()
        
        self.start_button.config(state = tk.DISABLED)
        self.stop_button.config(state = tk.NORMAL)

    def stop_recording(self):
        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None

        self.stop_event.set()
        self.record_thread.join()

        transcription = self.transcribe_audio(self.audio_filename)
        self.transcription_box.insert(
            tk.END,
            "Transcription:\n" + transcription + "\n"
        )
        
        speak_thread = threading.Thread(target = self.speak_audio, args = (agent_executor.run(transcription),))
        speak_thread.start()

        self.start_button.config(state = tk.NORMAL)
        self.stop_button.config(state = tk.DISABLED)

    def record_audio(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format = AUDIO_FORMAT,
            channels = CHANNELS,
            rate = FRAME_RATE,
            input = True,
            frames_per_buffer = CHUNK
        )

        while not self.stop_event.is_set():
            data = stream.read(CHUNK)
            self.audio_queue.put(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(self.audio_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(FRAME_RATE)
            wf.writeframes(b''.join(list(self.audio_queue.queue)))

    def transcribe_audio(self, filename):
        with open(filename, "rb") as audio_file:
            # transcript = openai.Audio.transcribe(
            #     model = "whisper-1",
            #     file = audio_file,
            #     language = "en"
            # )
            transcript = model.transcribe(filename)
            return transcript["text"].strip()

    def exit_program(self):
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AudioRecorderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

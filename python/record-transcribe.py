import openai
import os
import pyaudio
import pyttsx3
import threading
import tkinter as tk
import queue
import wave
import whisper
from tkinter import scrolledtext
from langchain import OpenAI, SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

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

        self.transcription_box = scrolledtext.ScrolledText(root, height = 10, width = 60)
        self.transcription_box.pack(padx = 10, pady = 10)

        self.recording_timer = None
        self.audio_filename = "audio_recording.wav"

    def start_recording(self):
        self.transcription_box.delete(1.0, tk.END)
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()

        self.record_thread = threading.Thread(target = self.record_audio)
        self.record_thread.start()

        self.recording_timer = self.root.after(20000, self.stop_recording)
        
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
            "Transcription:\n" + transcription + "\n" +
            "Result:\n" + agent_executor.run(transcription) + "\n"
        )

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

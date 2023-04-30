#!/usr/bin/env python
# coding: utf-8


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from keras import regularizers
import os

# In[27]:


# !/usr/bin/env python
# coding: utf-8

# In[ ]:


import sounddevice as sd
from tkinter import *
import queue
import soundfile as sf
import threading
from tkinter import messagebox

import pyaudio
import wave


#temp
import requests
# import lazy
# import resampy


#temp

# Define the user interface for Voice Recorder using Python
from tkinter import *
from PIL import ImageTk, Image

voice_rec = Tk()
voice_rec.geometry("1920x1080")
voice_rec.title("Speech Emotion Recognition")
# Create a queue to contain the audio data
q = queue.Queue()
# Declare variables and initialise them
recording = False
file_exists = False


# Fit data into queue
def callback(indata, frames, time, status):
    q.put(indata.copy())


# Functions to play, stop and record audio in Python voice recorder
# The recording is done as a thread to prevent it being the main process
def threading_rec(x):
    if x == 1:
        # If recording is selected, then the thread is activated
        t1 = threading.Thread(target=record_audio)
        t1.start()
    elif x == 2:
        # To stop, set the flag to false
        global recording
        recording = False
        messagebox.showinfo(message="Recording finished")
    elif x == 3:
        # To play a recording, it must exist.
        #        if file_exists:
        # Read the recording if it exists and play it
        data, fs = sf.read("audio.wav", dtype='float32')
        sd.play(data, fs)
        sd.wait()
    #        else:
    #            #Display and error if none is found
    #            messagebox.showerror(message="Record something to play")
    elif x == 4:

        from keras.models import model_from_json
        import tensorflow as tf
        # from tensorflow.keras.optimizers import RMSprop as opt
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.00001, epsilon=1e-6)
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("Emotion_Voice_Detection_Model.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        data, sampling_rate = librosa.load('audio.wav')
        import os
        import pandas as pd
        import glob

        # plt.figure(figsize=(15, 5))
        # librosa.display.waveplot(data, sr=sampling_rate)
        X, sample_rate = librosa.load('audio.wav', res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2 = pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        livedf2
        twodim = np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim,
                                         batch_size=32,
                                         verbose=1)
        # livepreds
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()

        print(liveabc)
        if liveabc == 0:
            print("Female_angry")
            Label(voice_rec, text="Female_angry", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 1:
            print("Female Calm")
            Label(voice_rec, text="Female Calm", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 2:
            print("Female Fearful")
            Label(voice_rec, text="Female Fearful", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 3:
            print("Female Happy")
            Label(voice_rec, text="Female Happy", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 4:
            print("Female Sad")
            Label(voice_rec, text="Female Sad", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 5:
            print("Male Angry")
            Label(voice_rec, text="Male Angry", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 6:
            print("Male calm")
            Label(voice_rec, text="Male calm", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 7:
            print("Male Fearful")
            Label(voice_rec, text="Male Fearful", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 8:
            print("Male Happy")
            Label(voice_rec, text="Male Happy", width=15, height=2, bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)
        elif liveabc == 9:
            print("Male sad")
            Label(voice_rec, text="Male sad", width=15, height=2,  bg="#6af8fc", fg="black", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=3)

        #SENDING DATA TO THINGSPEAK CHANNEL

        CHANNEL_ID = "2124840"
        WRITE_API_KEY = "TMOYUJPJ1KUESPVY"

        # URL for ThingSpeak
        BASE_URL = f"https://api.thingspeak.com/update?api_key=TMOYUJPJ1KUESPVY"

        # Data to be sent to ThingSpeak
        data = {"field1": liveabc, "field2": liveabc, "field3": liveabc}

        # Send data to ThingSpeak
        response = requests.post(BASE_URL, data=data)
        # Check the response status code
        if response.status_code == 200:
            print("Data sent to ThingSpeak successfully.")
        else:
            print("Failed to send data to ThingSpeak.")


# Recording function
def record_audio():
    # Declare global variables
    global recording
    # Set to True to record
    recording = True
    global file_exists
    # Create a file to save the audio
    messagebox.showinfo(message="Recording Audio. Speak into the mic")
    import pyaudio
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # paInt8
    CHANNELS = 2
    RATE = 44100  # sample rate
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)  # buffer

    print("* recording")
    Label(voice_rec, text="Recording started", width=15, height=2, bg="#7dfa64", fg="black", font=("Arial", 12, "bold")).grid(
        row=7, column=0, columnspan=3)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)  # 2 bytes(16 bits) per channel

    print("* done recording")
    Label(voice_rec, text="Recording finished", width=15, height=2, bg="#fa3c3c", fg="black", font=("Arial", 12, "bold")).grid(
        row=7, column=0, columnspan=3)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Load the image file
bg_image = Image.open("background_image.jpeg")
bg_image = bg_image.resize((1920, 1080))

bg_image = ImageTk.PhotoImage(bg_image)

# Create a label to display the image
bg_label = Label(voice_rec, image=bg_image)
bg_label.place(x=1, y=1, relwidth=1, relheight=1)

# title_lbl = Label(voice_rec, bg="white").grid(row=0, column=0, columnspan=3)

record_btn = Button(voice_rec, text="Record Audio", width=15, height=2, bg="#d6f26f", fg="black", font=("Arial", 12, "bold"), command=lambda m=1: threading_rec(m))
# Stop button
# stop_btn = Button(voice_rec, text="Stop Recording", width=15, height=2, bg="#d6f26f", fg="black", font=("Arial", 12, "bold"), command=lambda m=2: threading_rec(m))
# Play button
play_btn = Button(voice_rec, text="Play Recording", width=15, height=2, bg="#d6f26f", fg="black", font=("Arial", 12, "bold"), command=lambda m=3: threading_rec(m))
analyze_btn = Button(voice_rec, text="Analyze audio", width=15, height=2, bg="#d6f26f", fg="black", font=("Arial", 12, "bold"), command=lambda m=4: threading_rec(m))

# Add an empty row and column to center the buttons
# voice_rec.grid_rowconfigure(3, weight=1)
# voice_rec.grid_columnconfigure(1, weight=1)

# Position buttons
record_btn.grid(row=1, column=1, sticky="nswe", padx=700, pady=50)
# stop_btn.grid(row=1, column=1, sticky="nswe",padx=700, pady=50)
play_btn.grid(row=2, column=1, sticky="nswe",padx=700, pady=50)
analyze_btn.grid(row=3, column=1, sticky="nswe",padx=700, pady=50)

voice_rec.mainloop()

# In[ ]:
import warnings
import glob
import pdfplumber
import re
from csv import writer
import os

counter = 0
input_path = "abc"
pdf_files = glob.glob("%s/*.wav" % input_path)
for file in pdf_files:

    counter +=1
    # convert a WAV from stereo to mono
    from pydub import AudioSegment
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export(f"abc/{counter}.wav", format="wav")


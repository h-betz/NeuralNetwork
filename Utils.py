import numpy as np
import copy
import math
import ntpath
from scipy.fftpack import fft
from scipy.io import wavfile
from numpy.lib import stride_tricks
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import wave
import contextlib


def mel_Freq(file_name):

    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 20
    samples_per_frame = len(sig) / number_of_frames

    win_length = duration / number_of_frames

    i = 0
    x = 1
    frames = []
    frame_sample = []
    for n in range(0, number_of_frames):
        frame_sample = sig[i:(samples_per_frame*x)]
        i = samples_per_frame*x
        x += 1
        frames.append(frame_sample)


    mfcc_frames = []
    for frame in frames:
        mel = mfcc(frame, rate, win_length)
        mfcc_frames.append(mel)

    return mfcc_frames

# Get label associtaed with this file
def get_label(filename):
    ntpath.basename("letter_audio/speech/isolet1/fcmc0/*.wav")
    head, tail = ntpath.split(filename)
    start = tail.index('-')
    tail = tail[(start+1):]
    end = tail.index('-')
    fname = tail[0:end]
    return fname[0]

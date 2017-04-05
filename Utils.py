import numpy as np
import copy
import math
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
    number_of_frames = 40
    samples_per_frame = len(sig) / number_of_frames
    left = sig.T[0]

    i = 0
    x = 1
    frames = []
    frame_sample = []
    for n in range(0, number_of_frames):
        frame_sample = left[i:(samples_per_frame*x)]
        i = samples_per_frame*x
        x += 1
        frames.append(frame_sample)


    print(len(frames[1]))
    mfcc_frames = []
    for frame in frames:
        mel = mfcc(frame, rate)
        mfcc_frames.append(mel)

    print(len(mfcc_frames[0]))
    return mfcc_frames



def fourier(file_name):
    fs, data = wavfile.read(file_name)  # load the data
    a = data.T[0]  # this is a two channel soundtrack, I get the first track
    b = [(ele / 2 ** 8.) * 2 - 1 for ele in a]  # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b)  # calculate fourier transform (complex numbers list)
    # c = fft(data)
    d = len(c) / 2  # you only need half of the fft list (real signal symmetry)
    plt.plot(abs(c[:(d - 1)]), 'r')
    plt.xlabel("hz")
    plt.show()


def get_features(file_name):

    samplerate, samples = wavfile.read(file_name)
    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
    number_of_samples = len(samples)

    left = samples.T[0]

    samples_per_frame = number_of_samples / number_of_frames
    frames = []
    frame_samples = []
    i = 0
    for f in range(0, number_of_frames):
        frame_samples.append(left[i])
        i += 1
        for s in range(1, samples_per_frame):
            frame_samples.append(left[i])
            i += 1
        frames.append(frame_samples)
        frame_samples = []

    spectrum = []
    for frame in frames:
        spectrum.append(np.log10(np.abs(fft(frame))**2))


    w = np.fft.fft(left)
    freqs = np.fft.fftfreq(len(w))
    print(freqs.min(), freqs.max())

    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * rate)
    print(freq_in_hertz)
    #values = np.log10(np.abs(fft(left)))**2
    #print(values[10])
    #plt.specgram(values)

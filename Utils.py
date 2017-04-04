import numpy as np
import copy
from scipy.fftpack import fft
from scipy.io import wavfile
from numpy.lib import stride_tricks
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import wave
import contextlib


def mfcc(file_name):
    print(file_name)
    (rate, sig) = wavfile.read(file_name)
    mfcc_feat = mfcc(sig, rate, 1)
    print(len(mfcc_feat))
    return mfcc_feat



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


def stft(sig, framesize, overlapFrac=0.5, window=np.hanning):
    win = window(framesize)
    hopSize = int(framesize - np.floor(overlapFrac * framesize))

    #zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(framesize/2.0)), sig)
    #cols for windowing
    cols = np.ceil((len(samples) - framesize) / float(hopSize)) + 1
    #zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(framesize))

    frames = stride_tricks.as_strided(samples, shape=(cols, framesize), strides=(samples.strides[0]*hopSize, copy.copy(samples.strides[0])))
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))



def get_features(file_name):

    samplerate, samples = wavfile.read(file_name)
    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
    number_of_samples = len(samples)
    length = (duration * 1000)
    overlap = .5

    left = samples.T[0]
    #print(left[100])

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

    print(len(frames[2]))
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
from python_speech_features.sigproc import framesig
from python_speech_features.sigproc import powspec
from matplotlib.pyplot import specgram

import matplotlib.pyplot as plt
import wave
import contextlib


def mel_Freq(file_name):

    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
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


def get_features(file_name):
    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Frame our signal into 20 frames with 50% overlap
    number_of_frames = 40
    frame_len = len(sig) / (number_of_frames*(.5) + .5)
    frames = framesig(sig, frame_len, frame_len * .5)

    # A list of 20 frequency lists for each frame. 6 frequency bands with the average energy of each
    features = []
    band0 = []
    band1 = []
    band2 = []
    band3 = []
    band4 = []
    band5 = []
    for frame in frames:
        spectrum, freqs, t, img = specgram(frame, Fs=rate)
        i = 0
        bands = []
        for freq in freqs:
            if freq <= 400:
                band0.extend(spectrum[i])
            elif freq > 400 and freq <= 800:
                band1.extend(spectrum[i])
            elif freq > 800 and freq <= 1600:
                band2.extend(spectrum[i])
            elif freq > 1600 and freq <= 2800:
                band3.extend(spectrum[i])
            elif freq > 2800 and freq <= 4400:
                band4.extend(spectrum[i])
            elif freq > 4400:
                band5.extend(spectrum[i])
            i += 1
        bands.append(sum(band0) / len(band0))
        bands.append(sum(band1) / len(band1))
        bands.append(sum(band2) / len(band2))
        bands.append(sum(band3) / len(band3))
        bands.append(sum(band4) / len(band4))
        bands.append(sum(band5) / len(band5))
        features.append(bands)

    values = []
    for feature in features:
        for f in feature:
            values.append(f)

    return values



def get_mel(file_name):
    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
    frame_len = len(sig) / (number_of_frames * (.5) + .5)
    step = frame_len * .5
    frames = framesig(sig, frame_len, step)
    win_length = (duration * 1000) / number_of_frames

    mel_values = []
    for frame in frames:
        mel_values.append(mfcc(frame, rate, win_length))

    values = []
    for v in mel_values:
        for coefs in v:
            for coef in coefs:
                values.append(coef)
    return values
    # (rate, sig) = wavfile.read(file_name)
    #
    # with contextlib.closing(wave.open(file_name, 'r')) as f:
    #     frames = f.getnframes()
    #     rate = f.getframerate()
    #     duration = frames / float(rate)
    #
    # number_of_frames = 20
    # frame_len = len(sig) / (number_of_frames*(.5) + .5)
    # frames = framesig(sig, frame_len, frame_len * .5)
    #
    # win_length = (duration * 1000) / number_of_frames
    #
    # mel_values = []
    # for frame in frames:
    #     mel_values.append(mfcc(frame, rate, win_length))
    #
    # values = []
    # for v in mel_values:
    #     for coefs in v:
    #         for coef in coefs:
    #             values.append(coef)
    # return values


# def get_features(file_name):
#     (rate, sig) = wavfile.read(file_name)
#
#     with contextlib.closing(wave.open(file_name, 'r')) as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         duration = frames / float(rate)
#
#     # Frame our signal into 20 frames with 50% overlap
#     number_of_frames = 20
#     frame_len = len(sig) / (number_of_frames*(.5) + .5)
#     frames = framesig(sig, frame_len, frame_len * .5)
#
#     # A list of 20 frequency lists for each frame. 6 frequency bands with the average energy of each
#     features = []
#     band0 = []
#     band1 = []
#     band2 = []
#     band3 = []
#     band4 = []
#     band5 = []
#     for frame in frames:
#         spectrum, freqs, t, img = specgram(frame, Fs=rate)
#         i = 0
#         bands = []
#         for freq in freqs:
#             if freq <= 400:
#                 band0.extend(spectrum[i])
#             elif freq > 400 and freq <= 800:
#                 band1.extend(spectrum[i])
#             elif freq > 800 and freq <= 1600:
#                 band2.extend(spectrum[i])
#             elif freq > 1600 and freq <= 2800:
#                 band3.extend(spectrum[i])
#             elif freq > 2800 and freq <= 4400:
#                 band4.extend(spectrum[i])
#             elif freq > 4400:
#                 band5.extend(spectrum[i])
#             i += 1
#         bands.append(np.log2(sum(band0) / len(band0)))
#         bands.append(np.log2(sum(band1) / len(band1)))
#         bands.append(np.log2(sum(band2) / len(band2)))
#         bands.append(np.log2(sum(band3) / len(band3)))
#         bands.append(np.log2(sum(band4) / len(band4)))
#         bands.append(np.log2(sum(band5) / len(band5)))
#         #print(bands)
#         features.append(bands)
#
#     return features


# def get_features(file_name):
#
#     (rate, sig) = wavfile.read(file_name)
#
#     with contextlib.closing(wave.open(file_name, 'r')) as f:
#         frames = f.getnframes()
#         rate = f.getframerate()
#         duration = frames / float(rate)
#
#     number_of_frames = 15
#     win_length = duration / number_of_frames
#     frame_len = len(sig) / number_of_frames
#     frames = framesig(sig, frame_len, frame_len*.5)
#
#     mfcc_frames = []
#     for frame in frames:
#         mfcc_frames.append(mfcc(frame, rate, win_length))
#
#     return mfcc_frames[:29]


# Get label associtaed with this file
def get_label(filename):
    head, tail = ntpath.split(filename)
    start = tail.index('-')
    tail = tail[(start+1):]
    end = tail.index('-')
    fname = tail[0:end]
    return fname[0]

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def test(file_name):
    print(file_name)
    (rate, sig) = wav.read(file_name)
    mfcc_feat = mfcc(sig, rate, 1)
    print(len(mfcc_feat))
    return mfcc_feat


if __name__ == "__main__":
    #print(file_name)
    (rate, sig) = wav.read("original_audio/kss.wav")
    mfcc_feat = mfcc(sig, rate, 1)
    print(len(mfcc_feat))
    print(mfcc_feat[0,:])
    # val = test("shh.wav")
    # row = val[0, :]
    # plt.plot(row, 'r')
    # val = test("kss.wav")
    # row = val[0,:]
    # plt.plot(row, 'b')
    # plt.show()
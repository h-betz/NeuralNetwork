from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

def test():
    # (rate,sig) = wav.read("kss.wav")
    # mfcc_feat = mfcc(sig,rate)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig,rate)
    # print(len(fbank_feat))
    #print(fbank_feat[1:3,:])

    print("SHH:")
    (rate, sig) = wav.read("shh.wav")
    mfcc_feat = mfcc(sig, rate)
    print(len(mfcc_feat))
    #print(mfcc_feat[1:3,:])
    return mfcc_feat
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig, rate)
    # print(len(fbank_feat))
    # print(fbank_feat[1:3,:])

# if __name__ == "__main__":
#     test()


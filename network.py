import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import neuron
import mel

def get_fft(file_name):
    fs, data = wavfile.read(file_name) # load the data
    a = data.T[0] # this is a two channel soundtrack, I get the first track
    #b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    #c = fft(b) # calculate fourier transform (complex numbers list)
    c = fft(data)
    print(len(c))
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)
    plt.plot(abs(c[:]),'r')
    plt.show()

if __name__ == "__main__":
    #get_fft('kss.wav')
    #get_fft('shh.wav')
    mfcc = mel.test()
    count = 0
    for row in mfcc:
        print(count)
        for coef in row:
            print(coef)
        count += 1
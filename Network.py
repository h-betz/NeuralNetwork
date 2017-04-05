import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt


if __name__ == "__main__":

    #nng - 169
    #shh - 199
    #kss - 141
    #t, abs_fft_matrix = Utils.feature_extraction("audio/shh.wav")
    # plt.plot(t, abs_fft_matrix)
    # plt.ylabel('frequency')
    # plt.xlabel('time')
    # plt.show()
    #Utils.get_features("original_audio/nng.wav")
    Utils.mel_Freq("original_audio/shh.wav")
    Utils.mel_Freq("original_audio/kss.wav")
    Utils.mel_Freq("original_audio/nng.wav")
    #Utils.get_features("audio/shh.wav")
    #Utils.get_features("audio/kss.wav")

    #Build a layer of 40 input neurons
    input_layer = []
    i = 0
    while i < 40:
        n = Neuron.Neuron()
        input_layer.append(n)
        i += 1

    output_layer = []
    i = 0
    while i < 10:
        n = Neuron.Neuron()
        output_layer.append(n)
        i += 1
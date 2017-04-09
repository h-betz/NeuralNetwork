import numpy as np
import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt


if __name__ == "__main__":

    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 1000  # 100ms
    #Utils.mel_Freq("original_audio/shh.wav")
    #Utils.mel_Freq("original_audio/kss.wav")
    features = Utils.mel_Freq("original_audio/nng.wav")

    #Build a layer of 520 input neurons (40 frames, 13 features for each frame)
    input_layer = []
    for i in range(520):
        n = Neuron.Neuron()
        input_layer.append(n)

    #Build output layer, one neuron for each letter of the alphabet
    output_layer = []
    while i in range(26):
        n = Neuron.Neuron()
        output_layer.append(n)

    synapses = []

    #Feed features into our network
    i = 0
    for feature in features:
        for coef in feature[0]:
            n = input_layer[i]
            current = np.ones(time_ita) * coef
            time, v_plt, spike, tau = n.izh_simulation(a,b,c,d,time_ita, current, c)
            synapses.append(Synapse.Synapse(tau, time, spike))
            i += 1


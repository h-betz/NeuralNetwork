import numpy as np
import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt


def total_synaptic_value(neuron):
    conductance = 0
    for syn_k in neuron.synapses:
        output = syn_k.synapse(20)
        conductance += output
    return conductance


if __name__ == "__main__":

    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 1000  # 100ms
    features = Utils.mel_Freq("letter_audio/speech/isolet1/fcmc0/fcmc0-A1-t.wav")
    features_b = Utils.mel_Freq("letter_audio/speech/isolet1/fcmc0/fcmc0-B1-t.wav")

    #Build a layer of 240 input neurons (20 frames, 12 features for each frame)
    input_layer = []
    for i in range(240):
        n = Neuron.Neuron()
        input_layer.append(n)

    #Build output layer, one neuron for each letter of the alphabet
    output_layer = []
    for i in range(2):
        n = Neuron.Neuron()
        output_layer.append(n)

    synapses = []
    #Feed features into our network and get spike information (number of spikes, time of largest spike)
    i = 0
    for feature in features:
        for coefs in feature:
            coefs = iter(coefs)
            next(coefs)
            for coef in coefs:
                n = input_layer[i]
                current = np.ones(time_ita) * coef
                time, v_plt, spike, num_spikes, spike_times = n.izh_simulation(a,b,c,d,time_ita, current, c)
                synapses.append(Synapse.Synapse(time, spike, num_spikes, spike_times))
                i += 1

    # For each output neuron, append each of the synapses. This is important because the individual synapses are trained
    # for each neuron
    for out in output_layer:
        for syn in synapses:
            out.append_synapse(syn)

    for out in output_layer:
        for syn in out.synapses:
            if len(syn.spike_times) > 0:
                print(syn.spike_times)
        #current = np.ones(time_ita) * total_synaptic_value(out)
        #time, v_plt, spike, num_spikes, spike_times = out.izh_simulation(a,b,c,d,time_ita, current, c)


    plt.show()
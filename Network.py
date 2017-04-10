import numpy as np
import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt

# For some reason this isn't working...or at least doesn't appear to be
def total_conductance(neuron):
    conductance = 0
    for syn_k in neuron.synapses:
        t = 0
        for j in range(syn_k.spikes_received):
            conductance += (syn_k.conductance_amplitude * (t - syn_k.spike_times[j]) * np.exp(-(t - syn_k.spike_times[j])/syn_k.tau))
            t = syn_k.spike_times[j] - t

    return conductance


def total_synaptic_value(neuron):
    conductance = 0
    for syn_k in neuron.synapses:
        conductance += syn_k.spike
    return conductance


if __name__ == "__main__":

    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 1000  # 100ms
    features = Utils.mel_Freq("letter_audio/speech/isolet1/fcmc0/fcmc0-A1-t.wav")

    #Build a layer of 240 input neurons (20 frames, 12 features for each frame)
    input_layer = []
    for i in range(240):
        n = Neuron.Neuron()
        input_layer.append(n)

    #Build output layer, one neuron for each letter of the alphabet
    output_layer = []
    for i in range(26):
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
                time, v_plt, spike, tau, num_spikes, spike_times = n.izh_simulation(a,b,c,d,time_ita, current, c)
                synapses.append(Synapse.Synapse(tau, time, spike, num_spikes, spike_times))
                i += 1

    # For each output neuron, append each of the synapses. This is important because the individual synapses are trained
    # for each neuron
    for out in output_layer:
        for syn in synapses:
            out.append_synapse(syn)
        # for i in range(12):
        #     syn = synapses[(12*n)+i]
        #     syn_current = syn.synapse()
        # n += 1
        #print(total_conductance(out))


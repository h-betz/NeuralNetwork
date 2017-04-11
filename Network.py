import numpy as np
import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt

global input_layer
global output_layer
global synapses
global a, b, c, d, time_ita

class Network:

    def __init__(self):
        self.a = 0.02
        self.b = 0.2
        self.c = -65.
        self.d = 8.
        self.time_ita = 1000  # 100ms
        # Build a layer of 240 input neurons (20 frames, 12 features for each frame)
        self.input_layer = []
        for i in range(240):
            n = Neuron.Neuron()
            self.input_layer.append(n)

        self.synapses = []

        # Build output layer, one neuron for each letter of the alphabet
        self.output_layer = []
        for i in range(2):
            n = Neuron.Neuron()
            self.output_layer.append(n)


    def total_synaptic_value(self, neuron):
        conductance = 0
        for syn_k in neuron.synapses:
            output = syn_k.synapse(20)
            conductance += output
        return conductance


    def conduct_training(self, result):
        i = 0
        for out in output_layer:
            if i == result:
                for syn in out.synapses:
                    syn.Heb_STDP(out.pre_times, out.post_times)
            i += 1


    def start(self, fname):

        features = Utils.mel_Freq(fname)

        #Feed features into our network and get spike information (number of spikes, time of largest spike)
        i = 0
        for feature in features:
            for coefs in feature:
                coefs = iter(coefs)
                next(coefs)
                for coef in coefs:
                    n = self.input_layer[i]
                    current = np.ones(self.time_ita) * coef
                    time, v_plt, spike, num_spikes, spike_times = n.izh_simulation(self.a,self.b,self.c,self.d,self.time_ita, current, self.c)
                    self.synapses.append(Synapse.Synapse(time, spike, num_spikes, spike_times))
                    i += 1

        # For each output neuron, append each of the synapses. This is important because the individual synapses are trained
        # for each neuron
        i = 0
        for out in self.output_layer:
            for syn in self.synapses:
                out.append_synapse(syn)
                out.append_pre_times(syn.spike_times)
            self.output_layer[i] = out
            i += 1

        outputs = [0] * 2
        i = 0
        for out in self.output_layer:
            current = np.ones(self.time_ita) * self.total_synaptic_value(out)
            time, v_plt, spike, num_spikes, spike_times = out.izh_simulation(self.a,self.b,self.c,self.d,self.time_ita, current, self.c)
            out.append_post_times(spike_times)
            self.output_layer[i] = out
            if num_spikes > 0:
                outputs[i] = 1
            i += 1

        return outputs
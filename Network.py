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

        # Build a layer of 120 input neurons (10 frames, 12 features for each frame)
        self.input_layer = []
        for i in range(240):
            n = Neuron.Neuron()
            self.input_layer.append(n)

        # Build output layer, one neuron for each letter of the alphabet
        self.output_layer = []
        for i in range(2):
            n = Neuron.Neuron()
            self.output_layer.append(n)

        # For each input neuron, append one synapse to each output neuron
        for n in self.input_layer:
            for out in self.output_layer:
                synapse = Synapse.Synapse()
                synapse.set_out_neuron(out)
                synapse.set_input_neuron(n)
                n.append_synapse(synapse)
                out.append_synapse(synapse)


    # Get the total synaptic output for this neuron
    def total_synaptic_value(self, neuron):
        conductance = 0
        for syn_k in neuron.synapses:
            output = syn_k.synapse(20)
            conductance += output
        return conductance

    # if result == 0, then our target neuron is the first neuron in the output layer
    # result == 1 --> 2nd output neuron, result == 3 --> 3rd output neuron and so on
    def conduct_training(self, result):
        i = 0
        for out in self.output_layer:
            if i == result:
                # Undergo Hebbian STDP
                for syn in out.synapses:
                    syn.Heb_STDP()
            else:
                # Undergo anti-Hebbian STDP for non-target synapses
                for syn in out.synapses:
                    syn.Anti_Heb_STDP()
            i += 1

    # Perform analysis on the given filename
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

                    # Set pre spikes for each synapse connected to this neuron
                    for synapse in n.synapses:
                        synapse.set_pre_spikes(spike_times)
                        synapse.set_time(time)
                        synapse.set_spike(spike)
                    i += 1

        # Create a 2 neuron output vector
        outputs = [0] * 2
        i = 0
        for out in self.output_layer:
            current = np.ones(self.time_ita) * self.total_synaptic_value(out)
            time, v_plt, spike, num_spikes, spike_times = out.izh_simulation(self.a,self.b,self.c,self.d,self.time_ita, current, self.c)

            for syn in out.synapses:
                syn.set_post_spikes(spike_times)

            outputs[i] = num_spikes
            i += 1

        return outputs
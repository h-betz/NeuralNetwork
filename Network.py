import numpy as np
import Synapse
import Neuron
import Utils
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot

global input_layer
global output_layer
global hidden_layer
global synapses
global a, b, c, d, time_ita

class Network:

    def __init__(self, weights):
        # self.a = 0.03
        # self.b = -2
        # self.c = -50.
        # self.d = 100.
        self.a = 0.02
        self.b = 0.2
        self.c = -65.
        self.d = 8.
        self.time_ita = 3000  # 100ms

        # Build a layer of 120 input neurons (20 frames, 6 features for each frame)
        self.input_layer = []
        for i in range(520):
            n = Neuron.Neuron()
            self.input_layer.append(n)

        # Build output layer, one neuron for each letter of the alphabet
        self.output_layer = []
        for i in range(3):
            o = Neuron.Neuron()
            self.output_layer.append(o)

        i = 0
        # For each input neuron, append one synapse to each output neuron
        for n in self.input_layer:
            for out in self.output_layer:
                synapse = Synapse.Synapse()
                n.append_synapse(synapse)
                out.append_synapse(synapse)

        if weights is not None:
            i = 0
            n = 0
            s = 0
            with open(weights) as f:
                for line in f:
                    # For every 520 synapses, go to the next neuron
                    if i % 520 == 0 and n < len(self.output_layer):
                        s = 0
                        out = self.output_layer[n]
                        n += 1
                        out.synapses[s].set_weight(line)
                        s += 1
                    elif s < 520:
                        out.synapses[s].set_weight(line)
                        s += 1
                    i += 1

    # Calculates the current from the given MFCC value
    def get_current(self, x):
        if x == 0:
            return 8.5
        elif x < 0:
            return (.0530914398 * x) + 8.5
        else:
            return (.11805555555 * x) + 8.5
        # if x == 0:
        #     return 8.8
        # elif x < 0:
        #     return (.0949 * x) + 8.8
        # else:
        #     return (.1181 * x) + 8.8
        # if x == 0:
        #     return 4.7
        # elif x < 0:
        #     return (.0507 * x) + 4.7
        # else:
        #     return (.05 * x) + 4.7

    # Equation 8 from the paper
    def get_total(self, neuron):
        g_tot = 0
        for synapse in neuron.synapses:
            t = 0
            i = 0
            tau = 2
            #print("Synapse")
            for j in synapse.spike:
                if j == 1:
                    t_kj = synapse.time[i]
                    #g_tot += synapse.synapse(tau)
                    t = np.abs(t - t_kj)
                    #print("\t%s" % t)
                    g_tot += synapse.w * (t - t_kj) * np.exp(-(t - t_kj) / tau)
                    t = t_kj
                i += 1
        return g_tot


    # Get the total synaptic output for this neuron
    def total_synaptic_value(self, neuron):
        conductance = 0
        for syn_k in neuron.synapses:
            output = syn_k.synapse(2)
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


    # Perform analysis on the given filename using mel_Freq command
    def start(self, fname):

        features = Utils.get_mel(fname)
        features = features[:520]

        #Feed features into our network and get spike information (number of spikes, time of largest spike)
        i = 0

        # Use for mel_freq. 520 input neurons
        for feature in features:
            n = self.input_layer[i]
            #current = np.ones(self.time_ita) * feature
            current = np.ones(self.time_ita) * self.get_current(feature)
            time, v_plt, spike, num_spikes, spike_times = n.izh_simulation(self.a,self.b,self.c,self.d,self.time_ita, current, self.c)
            # plt.plot(time, v_plt, 'b-')
            # plt.show()
            # Set pre spikes for each synapse connected to this neuron
            for synapse in n.synapses:
                synapse.set_pre_spikes(spike_times)
                synapse.set_time(time)
                synapse.set_spike(spike)
            i += 1


        # Create a 3 neuron output vector
        outputs = [0] * 3
        spikes = []
        v_plts = []
        currents = []
        i = 0
        for out in self.output_layer:
            current = np.ones(self.time_ita) * self.total_synaptic_value(out)
            time, v_plt, spike, num_spikes, spike_times = out.output_izh_simulation(self.a,self.b,self.c,self.d,self.time_ita, current, self.c)
            spikes.append(spike_times)
            v_plts.append(v_plt)
            currents.append(current)
            for syn in out.synapses:
                syn.set_post_spikes(spike_times)

            outputs[i] = num_spikes
            i += 1

        return outputs, currents, time, v_plts, spikes
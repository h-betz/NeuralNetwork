import numpy as np
import random as random


class Synapse:

    global spike, time, conductance_amplitude, w, spikes_received, pre_spikes, post_spikes, input_neuron, out_neuron

    def __init__(self):
        self.post_spikes = []
        self.pre_spikes = []
        #self.w = random.uniform(0, 1)
        self.w = .5

    def set_weight(self, w):
        self.w = float(w)

    def set_input_neuron(self, neuron):
        self.input_neuron = neuron

    def set_out_neuron(self, neuron):
        self.out_neuron = neuron

    def set_time(self, time):
        self.time = time

    def set_spike(self, spike):
        self.spike = spike

    def set_pre_spikes(self, pre_spikes):
        self.pre_spikes = pre_spikes

    def set_post_spikes(self, post_spikes):
        self.post_spikes = post_spikes

    # Our W function for Hebbian STDP
    def synaptic_weight_func(self, delta_t):
        tau_pre = 20
        tau_post = 20
        Apre = .10
        Apost = -Apre
        if delta_t >= 0:
            return Apre*np.exp(-np.abs(delta_t)/tau_pre)
        if delta_t < 0:
            return Apost*np.exp(-np.abs(delta_t)/tau_post)

    # Our W function for anti-Hebbian STDP
    def anti_heb(self, delta_t):
        tau_pre = 20
        tau_post = 20
        Apre = .10
        Apost = -Apre
        if delta_t < 0:
            return Apre*np.exp(-np.abs(delta_t)/tau_pre)
        if delta_t >= 0:
            return Apost*np.exp(-np.abs(delta_t)/tau_post)

    # Same as Hebbian STDP except the cases are reversed
    def Anti_Heb_STDP(self):
        delta_w = 0
        for t_pre in self.pre_spikes:
            for t_post in self.post_spikes:
                delta_w += self.anti_heb(t_post - t_pre)
        # for t_pre in self.pre_spikes:
        #     for t_post in self.post_spikes:
        #         delta_w += self.anti_heb(t_post - t_pre)
        self.w += delta_w
        if self.w < 0:
            self. w = 0


    # Change in synaptic weight is the sum over all presynaptic spike times (t_pre) and postsynaptic spike times (t_post)
    # of some function W of the difference in these spike times
    def Heb_STDP(self):
        delta_w = 0
        # for t_pre in self.pre_spikes:
        #     for t_post in self.post_spikes:
        #         delta_w += self.anti_heb(t_post - t_pre)
        for t_pre in self.pre_spikes:
            for t_post in self.post_spikes:
                delta_w += self.synaptic_weight_func(t_post - t_pre)
        self.w += delta_w
        if self.w < 0:
            self.w = 0


    # Calculates synaptic output
    def synapse(self, tau):
        synapse_output = np.zeros(len(self.time))
        for t in range(len(self.time)):
            tmp_time = self.time[t] - self.time[0:t]
            synapse_output[t] = np.sum(((tmp_time * self.spike[0:t]) / tau) * np.exp(-(tmp_time * self.spike[0:t]) / tau))
        return self.w * synapse_output

    def synapse_func(self, tau):
        time = np.arange(10000) * 0.1
        func = time / tau * np.exp(-time / tau)
        return time, func
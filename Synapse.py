import numpy as np

class Synapse:

    global spike, time, conductance_amplitude, w, spikes_received, pre_spikes, post_spikes

    def __init__(self, time, spike, num_spikes, spike_times):
        self.spike = spike
        self.time = time
        self.conductance_amplitude = .7
        self.spikes_received = num_spikes
        self.pre_spikes = spike_times
        self.post_spikes = []
        self.w = .7


    def set_post_spikes(self, post_spikes):
        self.post_spikes = post_spikes

    def synaptic_weight_func(self, delta_t):
        tau_pre = tau_post = 20
        Apre = .01
        Apost = -Apre
        if delta_t >= 0:
            return Apre*np.exp(-delta_t/tau_pre)
        if delta_t < 0:
            return Apost*np.exp(delta_t/tau_post)

    def anti_heb(self, delta_t):
        tau_pre = tau_post = 20
        Apre = .01
        Apost = -Apre
        if delta_t <= 0:
            return Apre*np.exp(-delta_t/tau_pre)
        if delta_t > 0:
            return Apost*np.exp(delta_t/tau_post)


    def Anti_Heb_STDP(self):
        delta_w = 0
        for t_pre in self.pre_spikes:
            for t_post in self.post_spikes:
                delta_w += self.anti_heb(t_post - t_pre)
        self.w += delta_w
        print('Anti-Heb: ', self.w)


    def Heb_STDP(self):
        delta_w = 0
        for t_pre in self.pre_spikes:
            for t_post in self.post_spikes:
                delta_w += self.synaptic_weight_func(t_post - t_pre)
        self.w += delta_w
        print('Heb: ', self.w)


    def synapse(self, tau):
        synapse_output = np.zeros(len(self.time))
        for t in range(len(self.time)):
            tmp_time = self.time[t] - self.time[0:t]
            synapse_output[t] = np.sum(((tmp_time * self.spike[0:t]) / tau) * np.exp(-(tmp_time * self.spike[0:t]) / tau))
        return self.w * synapse_output

    # def synapse(self, tau, time, spike):
    #     synapse_output = np.zeros(len(time))
    #     for t in range(len(time)):
    #         tmp_time = time[t] - time[0:t]
    #         synapse_output[t] = np.sum(((tmp_time * spike[0:t]) / tau) * np.exp(-(tmp_time * spike[0:t]) / tau))
    #     return synapse_output

    def synapse_func(self, tau):
        time = np.arange(10000) * 0.1
        func = time / tau * np.exp(-time / tau)
        return time, func
import numpy as np

class Synapse:

    def __init__(self, time, spike, num_spikes, spike_times):
        self.spike = spike
        self.time = time
        self.conductance_amplitude = .7
        self.spikes_received = num_spikes
        self.spike_times = spike_times
        self.w = .7


    def synaptic_weight_func(self, delta_t):
        tau_pre = tau_post = 20
        Apre = 1.05
        Apost = -Apre
        if delta_t > 0:
            return Apre*np.exp(-delta_t/tau_pre)
        if delta_t < 0:
            return Apost*np.exp(delta_t/tau_post)

    def anti_heb(self, delta_t):
        tau_pre = tau_post = 20
        Apre = 1.05
        Apost = -Apre
        if delta_t < 0:
            return Apre*np.exp(-delta_t/tau_pre)
        if delta_t > 0:
            return Apost*np.exp(delta_t/tau_post)


    def Anti_Heb_STDP(self, t_pre, t_post):
        delta_w = 0
        for t_p in t_pre:
            for t_pst in t_post:
                delta_w += self.anti_heb(t_pst - t_p)
        return delta_w


    def Heb_STDP(self, t_pre, t_post):
        delta_w = 0
        for t_p in t_pre:
            for t_pst in t_post:
                delta_w += self.synaptic_weight_func(t_pst - t_p)
        return delta_w


    def synapse(self, tau):
        synapse_output = np.zeros(len(self.time))
        for t in range(len(self.time)):
            tmp_time = self.time[t] - self.time[0:t]
            synapse_output[t] = np.sum(((tmp_time * self.spike[0:t]) / tau) * np.exp(-(tmp_time * self.spike[0:t]) / tau))
        return self.conductance_amplitude * synapse_output

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
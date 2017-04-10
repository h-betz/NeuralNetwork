import numpy as np

class Synapse:

    def __init__(self, tau, time, spike, num_spikes, spike_times):
        self.spike = spike
        self.tau = tau
        self.time = time
        self.conductance_amplitude = .7
        self.spikes_received = num_spikes
        self.spike_times = spike_times

    def synapse(self):
        synapse_output = np.zeros(len(self.time))
        for t in range(len(self.time)):
            tmp_time = self.time[t] - self.time[0:t]
            synapse_output[t] = np.sum(((tmp_time * self.spike[0:t]) / self.tau) * np.exp(-(tmp_time * self.spike[0:t]) / self.tau))
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
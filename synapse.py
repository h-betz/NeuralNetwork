import numpy as np

class synapse:


    def synapse(self, tau, time, spike):
        synapse_output = np.zeros(len(time))
        for t in range(len(time)):
            tmp_time = time[t] - time[0:t]
            synapse_output[t] = np.sum(((tmp_time * spike[0:t]) / tau) * np.exp(-(tmp_time * spike[0:t]) / tau))
        return synapse_output

    def synapse_func(self, tau):
        time = np.arange(10000) * 0.1
        func = time / tau * np.exp(-time / tau)
        return time, func
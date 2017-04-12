import numpy as np
import random
import Synapse

global Pref, Pmin, Pth, D, Prest, pre_times, post_times, synapses
Pref = 0
Prest = 0
Pmin = -1
Pth = 5.5
D = 0.5


class Neuron:
    def __init__(self):
        self.Pth = Pth
        self.t_ref = 4
        self.t_rest = -1
        self.P = Prest
        self.D = D
        self.Pmin = Pmin
        self.Prest = Prest
        self.synapses = []
        self.post_times = []
        self.pre_times = []
        self.synapses = []

    def append_synapse(self, synapse):
        self.synapses.append(synapse)

    def append_pre_times(self, times):
        self.pre_times = times

    def append_post_times(self, times):
        self.post_times = times

    def append_synapse(self, synapse):
        self.synapses.append(synapse)

    def izh_simulation(self, a, b, c, d, time_ita, current, v_init):
        # a,b,c,d parameters for Izhikevich model
        # time_ita time iterations for euler method
        # current list of current for each time step
        # v_init initial voltage
        #tau = 0
        spike_times = []
        #max_spike = 0
        v = v_init
        #prev_max = v
        u = v * b
        v_plt = np.zeros(time_ita)
        u_plt = np.zeros(time_ita)
        spike = np.zeros(time_ita)
        num_spikes = 0
        tstep = 0.1  # ms
        ita = 0
        while ita < time_ita:
            v_plt[ita] = v
            u_plt[ita] = u
            v += tstep * (0.04 * (v ** 2) + 5 * v + 140 - u + current[ita])
            u += tstep * a * (b * v - u)
            if v > 30.:
                #max_spike = v
                spike[ita] = 1
                v = c
                u += d
                num_spikes += 1
                spike_times.append(ita)

            ita += 1
        time = np.arange(time_ita) * tstep
        #return time, v_plt, spike
        return time, v_plt, spike, num_spikes, spike_times

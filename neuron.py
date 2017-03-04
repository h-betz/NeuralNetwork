import matplotlib.pyplot as plt
import numpy as np
import math

def izh_simulation(a, b, c, d, time_ita, current, v_init):
    # a,b,c,d parameters for Izhikevich model
    # time_ita time iterations for euler method
    # current list of current for each time step
    # v_init initial voltage
    v = v_init
    u = v * b
    v_plt = np.zeros(time_ita)
    u_plt = np.zeros(time_ita)
    spike = np.zeros(time_ita)
    tstep = 0.1 #ms
    ita = 0
    while ita < time_ita:
        v_plt[ita] = v
        u_plt[ita] = u
        v += tstep * (0.04 * (v**2) + 5 * v + 140 - u + current[ita])
        u += tstep * a *(b * v - u)
        if v > 30.:
            spike[ita] = 1
            v = c
            u += d
        ita += 1
    time = np.arange(time_ita) * tstep
    return time, v_plt, spike

def synapse(tau, time, spike):
    synapse_output = np.zeros(len(time))
    for t in range(len(time)):
        tmp_time = time[t] - time[0:t]
        synapse_output[t] = np.sum(((tmp_time*spike[0:t])/tau) * np.exp(-(tmp_time*spike[0:t])/tau))
    return synapse_output

def synapse_func(tau):
    time = np.arange(10000) * 0.1
    func = time/tau * np.exp(-time/tau)
    return time, func

if __name__ == "__main__":
    # regular spiking
    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 10000 # 1s
    current = np.ones(time_ita) * 5.
    time1, v_plt1, spike1 = izh_simulation(a,b,c,d,time_ita, current, c)
    plt.figure(1)
    plt.plot(time1, v_plt1)
    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')
    plt.show()
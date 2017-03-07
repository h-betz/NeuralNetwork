import numpy as np
from neuron import neuron
from synapse import synapse
import matplotlib.pyplot as plt
import mel

if __name__ == "__main__":

    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 5000
    tau = 20
    mfcc = mel.test("shh.wav")
    # current = np.ones(time_ita) * mfcc[0,10]
    # time4, v_plt4, spike4 = izh_simulation(a, b, c, d, time_ita, current, c)
    # syn_output = synapse(tau, time4, spike4)
    # w = 10.
    # syn_current = w * syn_output
    # time5, v_plt5, spike5 = izh_simulation(a, b, c, d, time_ita, syn_current, c)
    #
    # plt.plot(time4, v_plt4, 'b-')
    # plt.plot(time5, v_plt5, 'g-')
    # plt.plot(time4, syn_current, 'r-')
    # plt.xlabel('time (ms)')
    # plt.ylabel('voltage (mV)')
    # plt.show()

    #Build our input layer
    input_layer = []
    i = 0
    while i < 13:
        n = neuron()
        input_layer.append(n)
        i += 1

    output_neuron = neuron()

    #Get spikes from our input layer and create a plot for each spike
    spikes = []
    times = []
    i = 0
    for coef in mfcc[0]:
        current = np.ones(time_ita) * coef
        neuron = input_layer[i]
        time, v_plt, spike = neuron.izh_simulation(a, b, c, d, time_ita, current, c)
        spikes.append(spike)
        times.append(time)
        plt.figure(i)
        plt.plot(time, v_plt)
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        i += 1

    #Get synapse from each input neuron
    synapses = []
    i = 0
    for spike in spikes:
        time = times[i]
        syn_output = synapse(tau, time, spike)
        w = 10.
        syn_current = w * syn_output
        synapses.append(syn_current)

    #I guess this is where I get stumped, 

    plt.show()

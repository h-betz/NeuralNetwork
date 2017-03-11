import numpy as np
from neuron import Neuron
from synapse import Synapse
import matplotlib.pyplot as plt
import mel

if __name__ == "__main__":

    a = 0.02
    b = 0.2
    c = -65.
    d = 8.
    time_ita = 5000
    tau = 20
    mfcc = mel.test("kss.wav")
    # current = np.ones(time_ita) * mfcc[0,6]
    # print(mfcc[0,10])
    # time4, v_plt4, spike4 = neuron().izh_simulation(a, b, c, d, time_ita, current, c)
    #syn_output = synapse(tau, time4, spike4)
    #w = 10.
    #syn_current = w * syn_output
    #time5, v_plt5, spike5 = neuron.izh_simulation(a, b, c, d, time_ita, syn_current, c)

    #plt.plot(time4, v_plt4, 'b-')
    #plt.plot(time5, v_plt5, 'g-')
    #plt.plot(time4, syn_current, 'r-')
    # plt.xlabel('time (ms)')
    # plt.ylabel('voltage (mV)')
    # plt.show()

    #Build our input layer
    input_layer = []
    i = 0
    while i < 12:
        n = Neuron()
        input_layer.append(n)
        i += 1

    output_neuron = Neuron()

    #Get spikes from our input layer and create a plot for each spike
    spikes = []
    times = []
    i = 0
    for coef in mfcc[0]:
        if i != 0:
            current = np.ones(time_ita) * coef
            n = input_layer[i-1]
            time, v_plt, spike = n.izh_simulation(a, b, c, d, time_ita, current, c)
            spikes.append(spike)
            times.append(time)
            # plt.figure(i)
            # plt.plot(time, v_plt)
            # plt.xlabel('time (ms)')
            # plt.ylabel('voltage (mV)')
        i += 1

    #Get synapse from each input neuron
    synapses = []
    i = 0
    for spike in spikes:
        time = times[i]
        syn_output = Synapse().synapse(tau, time, spike)
        w = 10.
        syn_current = w * syn_output
        synapses.append(syn_current)

    #Build hidden layer
    hidden_layer = []
    i = 0
    while i < 4:
        n = Neuron()
        hidden_layer.append(n)
        i += 1

    #Combine synapses
    hidden_synapses = []
    i = 0
    count = 0
    while i < 4:
        syn1 = synapses[i + count]
        count += 1
        syn2 = synapses[i + count]
        count += 1
        syn3 = synapses[i + count]
        lst = [x + y for x, y in zip(syn1, syn2)]
        hidden_synapses.append(lst)
        i += 1

    i = 0
    for syn in hidden_synapses:
        current = np.ones(time_ita) * syn
        n = hidden_layer[i]
        time, v_plt, spike = n.izh_simulation(a, b, c, d, time_ita, current, c)
        spikes.append(spike)
        times.append(time)
        plt.figure(i)
        plt.plot(time, v_plt)
        plt.xlabel('time (ms)')
        plt.ylabel('voltage (mV)')
        i += 1


    plt.show()

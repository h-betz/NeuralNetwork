from random import shuffle
import os
import sys
import Utils
import Network
import pyspike as spk
from pyspike import SpikeTrain
from datetime import datetime
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot

prototype_trains = [None] * 3

def write_weights(network):
    i = 0
    with open("weights.txt", "a") as f:
        for out in network.output_layer:
            if i == 0:
                f.write("A\n")
            elif i == 1:
                f.write("B\n")
            elif i == 2:
                f.write("C\n")
            elif i == 3:
                f.write("D\n")
            elif i == 4:
                f.write("E\n")
            elif i == 5:
                f.write("F\n")
            elif i == 6:
                f.write("G\n")
            elif i == 7:
                f.write("H\n")
            elif i == 8:
                f.write("I\n")
            elif i == 9:
                f.write("J\n")
            elif i == 10:
                f.write("K\n")
            elif i == 11:
                f.write("L\n")
            elif i == 12:
                f.write("M\n")
            elif i == 13:
                f.write("N\n")
            elif i == 14:
                f.write("O\n")
            elif i == 15:
                f.write("P\n")
            elif i == 16:
                f.write("Q\n")
            elif i == 17:
                f.write("R\n")
            elif i == 18:
                f.write("S\n")
            elif i == 19:
                f.write("T\n")
            elif i == 20:
                f.write("U\n")
            elif i == 21:
                f.write("V\n")
            elif i == 22:
                f.write("W\n")
            elif i == 23:
                f.write("X\n")
            elif i == 24:
                f.write("Y\n")
            elif i == 25:
                f.write("Z\n")
            for syn in out.synapses:
                if i == 0:
                    f.write("%s\n" % syn.w)
                    i = 1
                else:
                    f.write("%s\n" % syn.w)
                    i = 0
            i += 1

def print_result(results):
    print('\tA: ' + str(results[0]))
    print('\tB: ' + str(results[1]))
    print('\tX: ' + str(results[2]))
    print('\tD: ' + str(results[3]))
    print('\tE: ' + str(results[4]))
    print('\tF: ' + str(results[5]))
    print('\tG: ' + str(results[6]))
    print('\tH: ' + str(results[7]))
    print('\tI: ' + str(results[8]))
    print('\tJ: ' + str(results[9]))
    print('\tK: ' + str(results[10]))
    print('\tL: ' + str(results[11]))
    print('\tM: ' + str(results[12]))
    print('\tN: ' + str(results[13]))
    print('\tO: ' + str(results[14]))
    print('\tP: ' + str(results[15]))
    print('\tQ: ' + str(results[16]))
    print('\tR: ' + str(results[17]))
    print('\tS: ' + str(results[18]))
    print('\tT: ' + str(results[19]))
    print('\tU: ' + str(results[20]))
    print('\tV: ' + str(results[21]))
    print('\tW: ' + str(results[22]))
    print('\tX: ' + str(results[23]))
    print('\tY: ' + str(results[24]))
    print('\tZ: ' + str(results[25]))

# Generate a spike train from the given spike
def generate_prototypes(spike, key):
    global prototype_trains
    spike_train = SpikeTrain(spike, [0.0, 300.0])
    if key == 'A':
        prototype_trains[0] = spike_train
    elif key == 'B':
        prototype_trains[1] = spike_train
    elif key == 'X':
        prototype_trains[2] = spike_train

def spike_analysis(spikes, value):
    distances = []
    i = 0
    for spike in spikes:
        spike_train = SpikeTrain(spike, [0.0, 300.0])
        isi_profile = spk.spike_sync(prototype_trains[i], spike_train)
        distances.append(isi_profile)
        i += 1

    val, idx = max((val, idx) for (idx, val) in enumerate(distances))
    print("Distance: %.8f" % val)
    print("Index: %s" % idx)


def show_plots(time, v_plts, currents, spikes):
    plt.figure('A')
    plt.plot(time, v_plts[0], 'g-')
    plt.plot(time, currents[0], 'r-')
    plt.figure('B')
    plt.plot(time, v_plts[1], 'b-')
    plt.plot(time, currents[1], 'y-')
    plt.figure('X')
    plt.plot(time, v_plts[2], 'k-')
    plt.plot(time, currents[2], 'm-')
    plt.figure('All')
    plt.plot(time, v_plts[0], 'g-')
    plt.plot(time, currents[0], 'r-')
    plt.plot(time, v_plts[1], 'b-')
    plt.plot(time, currents[1], 'y-')
    plt.plot(time, v_plts[2], 'k-')
    plt.plot(time, currents[2], 'm-')
    sp = spikeplot.SpikePlot()
    sp.plot_spikes(spikes)
    plt.show()

# Test our network
def test():
    global prototype_trains
    prototype_trains = [None] * 3
    mapping = dict()

    weights = "weights.txt"

    network = Network.Network(weights=weights)
    audio_path = "letter_audio/speech/isolet3"

    audio = [os.path.join(root, name)
                 for root, dirs, files in os.walk(audio_path)
                 for name in files
                 if name.endswith((".wav"))]

    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

    a_count = 1
    b_count = 1
    x_count = 1
    count = 3
    for key in mapping:
        if mapping[key] == 'A' and a_count != 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            a_count -= 1
            count -= 1
            if a_count == 0:
                # Generate a spike train for the 'A' sound
                generate_prototypes(spikes[0], 'A')
            elif a_count == 0 and b_count == 0 and x_count == 0:
                spike_analysis(spikes)
                #show_plots(time, v_plts, currents, spikes)
                #spike_analysis(spikes)
        elif mapping[key] == 'B' and b_count != 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            b_count -= 1
            count -= 1
            if b_count == 0:
                # Generate a spike train for the 'B' sound
                generate_prototypes(spikes[1], 'B')
            elif a_count == 0 and b_count == 0 and x_count == 0:
                spike_analysis(spikes)
                #show_plots(time, v_plts, currents, spikes)
                #spike_analysis(spikes)
        elif mapping[key] == 'X' and x_count != 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            x_count -= 1
            count -= 1
            if x_count == 0:
                # Generate a spike train for the 'X' sound
                generate_prototypes(spikes[2], 'X')
            elif a_count == 0 and b_count == 0 and x_count == 0:
                spike_analysis(spikes)
        elif count == 0:
            print(key)
            value = ''
            if mapping[key] == 'A':
                value = 'A'
            elif mapping[key] == 'B':
                value = 'B'
            elif mapping[key] == 'X':
                value = 'X'
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            spike_analysis(spikes, value)


# Train the network
def train():
    network = Network.Network(weights=None)

    mapping = dict()

    audio_path = "letter_audio/speech/isolet1"

    # Gets list of all audio files in the directory
    audio = [os.path.join(root, name)
             for root, dirs, files in os.walk(audio_path)
             for name in files
             if name.endswith((".wav"))]

    audio_path = "letter_audio/speech/isolet2"
    audio2 = [os.path.join(root, name)
              for root, dirs, files in os.walk(audio_path)
              for name in files
              if name.endswith((".wav"))]

    audio.extend(audio2)

    shuffle(audio)

    # Get a mapping of labels to audio
    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

    print(datetime.now())

    a_count = 20
    b_count = 20
    c_count = 20
    d_count = 20
    e_count = 20
    f_count = 20
    g_count = 20
    h_count = 20
    i_count = 20
    j_count = 20
    k_count = 20
    l_count = 20
    m_count = 20
    n_count = 20
    o_count = 20
    p_count = 20
    q_count = 20
    r_count = 20
    s_count = 20
    t_count = 20
    u_count = 20
    v_count = 20
    w_count = 20
    x_count = 20
    y_count = 20
    z_count = 20

    for key in mapping:
        if mapping[key] == 'A' and a_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(0)
            a_count -= 1
        elif mapping[key] == 'B' and b_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(1)
            b_count -= 1
        elif mapping[key] == 'C' and c_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(2)
            c_count -= 1
        elif mapping[key] == 'D' and d_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(3)
            d_count -= 1
        elif mapping[key] == 'E' and e_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(4)
            e_count -= 1
        elif mapping[key] == 'F' and f_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(5)
            f_count -= 1
        elif mapping[key] == 'G' and g_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(6)
            g_count -= 1
        elif mapping[key] == 'H' and h_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(7)
            h_count -= 1
        elif mapping[key] == 'I' and i_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(8)
            i_count -= 1
        elif mapping[key] == 'J' and j_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(9)
            j_count -= 1
        elif mapping[key] == 'K' and k_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(10)
            k_count -= 1
        elif mapping[key] == 'L' and l_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(11)
            l_count -= 1
        elif mapping[key] == 'M' and m_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(12)
            m_count -= 1
        elif mapping[key] == 'N' and n_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(13)
            n_count -= 1
        elif mapping[key] == 'O' and o_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(14)
            o_count -= 1
        elif mapping[key] == 'P' and p_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(15)
            p_count -= 1
        elif mapping[key] == 'Q' and q_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(16)
            q_count -= 1
        elif mapping[key] == 'R' and r_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(17)
            r_count -= 1
        elif mapping[key] == 'S' and s_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(18)
            s_count -= 1
        elif mapping[key] == 'T' and t_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(19)
            t_count -= 1
        elif mapping[key] == 'U' and u_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(20)
            u_count -= 1
        elif mapping[key] == 'V' and v_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(21)
            v_count -= 1
        elif mapping[key] == 'W' and w_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(22)
            w_count -= 1
        elif mapping[key] == 'X' and x_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(23)
            x_count -= 1
        elif mapping[key] == 'Y' and y_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(24)
            y_count -= 1
        elif mapping[key] == 'Z' and z_count > 0:
            print(key)
            results = network.start(key)
            print_result(results)
            network.conduct_training(25)
            z_count -= 1

    write_weights(network)
    print(datetime.now())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print('Training...')
            train()
        else:
            print('Testing')
            test()



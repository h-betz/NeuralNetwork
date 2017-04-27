from random import shuffle
import os
import Utils
import Network
from datetime import datetime
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot

def write_weights():
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
    #print('\tX: ' + str(results[2]))
    # print('\tD: ' + str(results[3]))
    # print('\tE: ' + str(results[4]))
    # print('\tF: ' + str(results[5]))
    # print('\tG: ' + str(results[6]))
    # print('\tH: ' + str(results[7]))
    # print('\tI: ' + str(results[8]))
    # print('\tJ: ' + str(results[9]))
    # print('\tK: ' + str(results[10]))
    # print('\tL: ' + str(results[11]))
    # print('\tM: ' + str(results[12]))
    # print('\tN: ' + str(results[13]))
    # print('\tO: ' + str(results[14]))
    # print('\tP: ' + str(results[15]))
    # print('\tQ: ' + str(results[16]))
    # print('\tR: ' + str(results[17]))
    # print('\tS: ' + str(results[18]))
    # print('\tT: ' + str(results[19]))
    # print('\tU: ' + str(results[20]))
    # print('\tV: ' + str(results[21]))
    # print('\tW: ' + str(results[22]))
    # print('\tX: ' + str(results[23]))
    # print('\tY: ' + str(results[24]))
    # print('\tZ: ' + str(results[25]))


if __name__ == "__main__":

    network = Network.Network()

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
    # audio_path = "letter_audio/speech/isolet3"
    # audio3 = [os.path.join(root, name)
    #              for root, dirs, files in os.walk(audio_path)
    #              for name in files
    #              if name.endswith((".wav"))]
    # audio.extend(audio3)

    shuffle(audio)


    # audio_path = "letter_audio/speech/isolet4"
    # audio4 = [os.path.join(root, name)
    #              for root, dirs, files in os.walk(audio_path)
    #              for name in files
    #              if name.endswith((".wav"))]
    # audio.extend(audio4)

    # Get a mapping of labels to audio
    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

    print(datetime.now())
    count = 30
    #x_count = 15
    a_count = 35
    b_count = 35
    for key in mapping:
        if mapping[key] == 'A' and a_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(0)
            a_count -= 1
            if a_count == 0 and b_count == 0:
                plt.figure('A')
                plt.plot(time, v_plts[0], 'g-')
                plt.plot(time, currents[0], 'r-')
                plt.figure('B')
                plt.plot(time, v_plts[1], 'b-')
                plt.plot(time, currents[1], 'y-')
                # plt.figure('X')
                # plt.plot(time, v_plts[2], 'k-')
                # plt.plot(time, currents[2], 'm-')
                sp = spikeplot.SpikePlot()
                sp.plot_spikes(spikes)
                plt.show()
                write_weights()
        elif mapping[key] == 'B' and b_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(1)
            b_count -= 1
            if a_count == 0 and b_count == 0:
                plt.figure('A')
                plt.plot(time, v_plts[0], 'g-')
                plt.plot(time, currents[0], 'r-')
                plt.figure('B')
                plt.plot(time, v_plts[1], 'b-')
                plt.plot(time, currents[1], 'y-')
                # plt.figure('X')
                # plt.plot(time, v_plts[2], 'k-')
                # plt.plot(time, currents[2], 'm-')
                sp = spikeplot.SpikePlot()
                sp.plot_spikes(spikes)
                plt.show()
                write_weights()
        # elif mapping[key] == 'X' and x_count > 0:
        #     print(key)
        #     results, currents, time, v_plts, spikes = network.start(key)
        #     print_result(results)
        #     network.conduct_training(2)
        #     x_count -= 1
        #     if x_count == 0 and a_count == 0 and b_count == 0:
        #         plt.figure('A')
        #         plt.plot(time, v_plts[0], 'g-')
        #         plt.plot(time, currents[0], 'r-')
        #         plt.figure('B')
        #         plt.plot(time, v_plts[1], 'b-')
        #         plt.plot(time, currents[1], 'y-')
        #         plt.figure('X')
        #         plt.plot(time, v_plts[2], 'k-')
        #         plt.plot(time, currents[2], 'm-')
        #         sp = spikeplot.SpikePlot()
        #         sp.plot_spikes(spikes)
        #         plt.show()
        #         write_weights()
        # elif mapping[key] == 'D':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(3)
        # elif mapping[key] == 'E':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(4)
        # elif mapping[key] == 'F':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(5)
        # elif mapping[key] == 'G':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(6)
        # elif mapping[key] == 'H':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(7)
        # elif mapping[key] == 'I':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(8)
        # elif mapping[key] == 'J':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(9)
        # elif mapping[key] == 'K':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(10)
        # elif mapping[key] == 'L':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(11)
        # elif mapping[key] == 'M':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(12)
        # elif mapping[key] == 'N':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(13)
        # elif mapping[key] == 'O':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(14)
        # elif mapping[key] == 'P':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(15)
        # elif mapping[key] == 'Q':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(16)
        # elif mapping[key] == 'R':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(17)
        # elif mapping[key] == 'S':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(18)
        # elif mapping[key] == 'T':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(19)
        # elif mapping[key] == 'U':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(20)
        # elif mapping[key] == 'V':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(21)
        # elif mapping[key] == 'W':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(22)
        # elif mapping[key] == 'X':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(23)
        # elif mapping[key] == 'Y':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(24)
        # elif mapping[key] == 'Z':
        #     print(key)
        #     results = network.start(key)
        #     print_result(results)
        #     network.conduct_training(25)

    print(datetime.now())



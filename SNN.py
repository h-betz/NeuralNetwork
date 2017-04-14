import glob
import os
import Utils
import Network
from matplotlib import pyplot as plt

if __name__ == "__main__":

    network = Network.Network()

    mapping = dict()

    audio_path = "letter_audio/speech/isolet1"

    # Gets list of all audio files in the directory
    audio = [os.path.join(root, name)
                 for root, dirs, files in os.walk(audio_path)
                 for name in files
                 if name.endswith((".wav"))]


    # Get a mapping of labels to audio
    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

    for key in mapping:
        if mapping[key] == 'H':
            print(mapping[key])
            results = network.start(key)
            print('\tH: ' + str(results[0]))
            print('\tF: ' + str(results[1]))
            network.conduct_training(0)
        elif mapping[key] == 'F':
            print(mapping[key])
            results = network.start(key)
            print('\tH: ' + str(results[0]))
            print('\tF: ' + str(results[1]))
            network.conduct_training(1)



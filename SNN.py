import glob
import Utils
import Network

if __name__ == "__main__":

    network = Network.Network()

    mapping = dict()

    audio_path = "letter_audio/speech/isolet1/fcmc0/*.wav"

    # Get a mapping of labels to audio
    for fname in glob.glob(audio_path):
        mapping[fname] = Utils.get_label(fname)

    for key in mapping:
        if mapping[key] == 'A':
            results = network.start(key)
            #network.conduct_training(0)

        elif mapping[key] == 'B':
            results = network.start(key)


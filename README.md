# NeuralNetwork
Spiking Neural Network to identify spoken letters

With the emergence of intelligent personal assistants that primarily rely on voice input,
such as Amazon’s Alexa and Google Assistant, there is increased demand for speech recognition
systems. Being able to quickly and efficiently recognize spoken letters is the first step in
developing such a system. Our goal is to use a spiking neural network (SNN) to accurately
classify spoken stop consonants. The network utilized Izhikevich regularly spiking neurons and
spike-timing dependent plasticity to learn from a database of training data. Our SNN generated
distinct spiking signatures for each of the spoken stop consonants, but ultimately failed to
accurately classify the consonants based on these signatures. This result was not due to the
network architecture, but was likely caused by the equation used to calculate the injected current.
We assume that further investigation would yield better results, however our study was a step in
the right direction

Speech recognition has become increasingly popular; as a result, the need for accurate
and fast speech recognition systems is rising in demand. Using neural networks for speech
recognition is not new and past studies have used various neural networks for their models.
Using a Spiking Neural Network (SNN), it was determined that the English stop consonants (b,
d, g, k, t, p) produce distinct spiking patterns. These distinct patterns allow the spoken
consonants to be identified and classified, which can be useful in many speech recognition
systems such as Amazon’s Alexa, Google Assistant, or Microsoft’s Cortana.
Inspiration for this study came from an article published in Neurocomputing by
Amirhossein Tavanaei and Anthony Maida from the Center for Advanced Computer Studies at
the University of Louisiana [1]. In this study, the authors used an SNN to generate distinct spike
signatures for the spoken digits 0 – 9 and to classify the spoken digit based off these signatures.
We decided to take a similar approach but apply it to the English stop consonants which had
been done in a separate study using a Time Delay Neural Network (TDNN). By successfully
achieving our goal in this project, we hope to expand the list of letters that can be recognized to
include the entire English alphabet.

To Train:
1. Navigate to the directory containing SNN.py 
2. run the command python SNN.py train

To Test:
1. Navigate to the directory containing SNN.py
2. run the command python SNN.py

"""Test module"""
import os

import matplotlib.pyplot as plt
import numpy as np

from key_gen import key_generator
from scrambler import scramble_audio_file

# Constants #######################################################################################

# Flag for controlling if plots should be created to visualize the shuffling of frequencies
PLOT_DEBUG = True

# Amount of shuffling operations to be done
NUMBER_OF_SHUFFLING_ITERATIONS = int(1e4)

# Bandwidth of each frequency band to be shuffled with each other, given in kHz
BANDWIDTH = 1

# The upper limit frequency to be shuffled, given in kHz
MAX_SHUFFLING_FREQUENCY = 15

# Main script #####################################################################################

# Generate encryption key from user input - scrambling
key_scram = key_generator('scrambling', NUMBER_OF_SHUFFLING_ITERATIONS, BANDWIDTH, MAX_SHUFFLING_FREQUENCY)

# Use encryption as instructions for the frequency scrambling function
r, _ = scramble_audio_file(key_scram, BANDWIDTH, 'scrambling', PLOT_DEBUG)

# Generate encryption key from user input - de-scrambling
key_descram = key_generator('de-scrambling', NUMBER_OF_SHUFFLING_ITERATIONS, BANDWIDTH, MAX_SHUFFLING_FREQUENCY)

# Use encryption as instructions for the frequency de-scrambling function
_, s = scramble_audio_file(key_descram, BANDWIDTH, 'de-scrambling', PLOT_DEBUG)

list_index = 0
key_len = len(key_descram)

# Test that keys are equal
while list_index < key_len:

    es_b1 = key_scram[list_index]['b1_cfreq']
    es_b2 = key_scram[list_index]['b2_cfreq']

    esd_b1 = key_descram[key_len-list_index-1]['b1_cfreq']
    esd_b2 = key_descram[key_len-list_index-1]['b2_cfreq']

    if es_b1 != esd_b1:
        print('not equal - b1')
        print(es_b1, esd_b1)

    if es_b2 != esd_b2:
        print('not equal - b2')
        print(es_b2, esd_b2)

    list_index += 1

# Show all the created plots if plot debug is enabled
if PLOT_DEBUG:
    a = []
    plt.figure()
    for i in range(len(r)-1):
        a.append((1/len(s) * np.abs(s[i]))-(1/len(r) * np.abs(r[i])))
    plt.plot(a)
    plt.show()

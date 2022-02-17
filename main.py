"""Main module"""
import os

import matplotlib.pyplot as plt

from key_gen import key_generator
from scrambler import scramble_audio_file

# Constants #######################################################################################

# Flag for controlling if plots should be created to visualize the shuffling of frequencies
PLOT_DEBUG = True

# Amount of shuffling operations to be done
NUMBER_OF_SHUFFLING_ITERATIONS = int(1e2)

# Bandwidth of each frequency band to be shuffled with each other, given in kHz
BANDWIDTH = 1

# The upper limit frequency to be shuffled, given in kHz
MAX_SHUFFLING_FREQUENCY = 15

# Determine if a scrambling or de-scrambling oaperation is needed and set flag accordingly
dir_entries = os.listdir(os.getcwd())

# Determination of operation type #################################################################
if "Scrambled_audio.wav" in dir_entries:
    type_of_scrambling = 'de-scrambling'
elif "Recording.wav" in dir_entries:
    type_of_scrambling = 'scrambling'
else:
    # We need a recording to start with, so we raise and error
    raise FileNotFoundError("No recordings found in current directory - Please add one")

# Main script #####################################################################################

# Generate encryption key from user input
key = key_generator(type_of_scrambling, NUMBER_OF_SHUFFLING_ITERATIONS, BANDWIDTH, MAX_SHUFFLING_FREQUENCY)

# Use encryption as instructions for the frequency scrambling function
scramble_audio_file(key, BANDWIDTH, type_of_scrambling, PLOT_DEBUG)

# Show all the created plots if plot debug is enabled
if PLOT_DEBUG:
    plt.show()

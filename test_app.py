"""Test module"""
import pytest

import matplotlib.pyplot as plt
import numpy as np

from key_gen import *
from scrambler import scramble_audio_file


class FixtureOutput():
    """Object containging all parameters of interest from a full scramble to de-scramble operation.

    This object is used to make it easier to pass all relevant parameters from the test fixture.
    """

    def __init__(self, original_freq, de_scrambled_freq, key_scram, key_descram):
        self.original_freq = original_freq
        self.de_scrambled_freq = de_scrambled_freq
        self.key_scram = key_scram
        self.key_descram = key_descram


# Arrange test parameters
@pytest.fixture
def init_test_scrambling():
    """Pytest fixture used to setup a scrambled and de_scrambled signal together with their respective keys.

    Args:
        None

    Return:
        FixtureOutput(FixtureOutput): FixtureOutput object containing all information from the entire scrambling
                                      and descramling operation.
    """

    # Constants ########################################################################################

    # Flag for controlling if plots should be created to visualize the shuffling of frequencies
    PLOT_DEBUG = False

    # Amount of shuffling operations to be done
    NUMBER_OF_SHUFFLING_ITERATIONS = int(1e3)

    # Bandwidth of each frequency band to be shuffled with each other, given in kHz
    BANDWIDTH = 1

    # The upper limit frequency to be shuffled, given in kHz
    MAX_SHUFFLING_FREQUENCY = 15

    # Hardcoded password used for testing to avoid asking for user input
    DEBUG_PASSWORD = 'password'

    # Generate keys and signals ########################################################################

    # Generate encryption key from user input - scrambling
    key_scram = key_generator('scrambling', NUMBER_OF_SHUFFLING_ITERATIONS,
                              BANDWIDTH, MAX_SHUFFLING_FREQUENCY, DEBUG_PASSWORD)

    # Use encryption as instructions for the frequency scrambling function
    original_freq, _ = scramble_audio_file(key_scram, BANDWIDTH, 'scrambling', PLOT_DEBUG)

    # Generate encryption key from user input - de-scrambling
    key_descram = key_generator('de-scrambling', NUMBER_OF_SHUFFLING_ITERATIONS,
                                BANDWIDTH, MAX_SHUFFLING_FREQUENCY, DEBUG_PASSWORD)

    # Use encryption as instructions for the frequency de-scrambling function
    _, de_scrambled_freq = scramble_audio_file(key_descram, BANDWIDTH, 'de-scrambling', PLOT_DEBUG)

    return FixtureOutput(original_freq=original_freq,
                         de_scrambled_freq=de_scrambled_freq,
                         key_scram=key_scram,
                         key_descram=key_descram)


def test_key_equal(init_test_scrambling):
    """Test if scrambling and de-scrambling keys are equal."""
    list_index = 0
    key_len = len(init_test_scrambling.key_descram)

    # Test that keys are equal
    while list_index < key_len:

        # Get scrambling frequency pair for specific list_index
        es_b1 = init_test_scrambling.key_scram[list_index]['b1_cfreq']
        es_b2 = init_test_scrambling.key_scram[list_index]['b2_cfreq']

        # Get de-scrambling frequency pair for specific list_index
        esd_b1 = init_test_scrambling.key_descram[key_len-list_index-1]['b1_cfreq']
        esd_b2 = init_test_scrambling.key_descram[key_len-list_index-1]['b2_cfreq']

        # They should be equal for all parts of the encryption key
        assert es_b1 == esd_b1
        assert es_b2 == esd_b2

        list_index += 1


def test_signal_recognizability(init_test_scrambling):
    """Test induced signal noise from a full cycle of scrambling and de-scrambling."""
    freq_differnce = []

    for i in range(len(init_test_scrambling.original_freq)-1):
        de_scram_freq_amp = 1/len(init_test_scrambling.de_scrambled_freq) * \
            np.abs(init_test_scrambling.de_scrambled_freq[i])
        original_signal_freq_amp = 1/len(init_test_scrambling.original_freq) * \
            np.abs(init_test_scrambling.original_freq[i])
        freq_differnce.append(de_scram_freq_amp-original_signal_freq_amp)

    # Plot difference to visualize error
    # TODO: Delete this plot when the assert is added to the test
    plt.figure()
    plt.plot(freq_differnce)
    plt.show()

    # TODO: Add assert on the lowest allowed difference between the input and de-scrambled signal

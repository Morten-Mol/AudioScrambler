"""Code concerning the scrambling of the frequency bands of an audio file."""

import wave
import os
from array import array
from time import time
from copy import deepcopy

import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt


def scramble_audio_file(key, w, type_of_scrambling, plot_debug):
    """Scrambles frequency bands of an audio file.

    Args:
        key (list(dicts)): List of dictionaries, each containing a set of frequencies to be shuffled.
        w (float): Bandwidth of the spectra to be shuffled, given in kHz
        type_of_scrambling (str): Flag to control if the action needed is 'scrambling' or 'de-scrambling'
        plot_debug (bool): Flag to control whether plots of the shuffling are to be created.

    Returns:
        None
    """
    # Import audio file
    if type_of_scrambling == 'scrambling':
        fn_start = 'Recording.wav'
    else:
        fn_start = 'Scrambled_audio.wav'

    filename = os.path.join(os.getcwd(), fn_start)

    # Check for the audio sample size, given in bytes
    with wave.open(filename, mode='rb') as wav:
        nchan, sample_width, sample_freq, total_samples, _, _ = wav.getparams()
        print(total_samples)

    possible_sample_sizes = {1: 'B', 2: 'h', 4: 'i'}
    audio_sample_size = possible_sample_sizes[sample_width]

    # Create array with the same type as the audio sample size
    raw = array(audio_sample_size)
    raw.fromfile(open(filename, 'rb'), int(total_samples))
    # int(os.path.getsize(filename)/raw.itemsize)

    # Clean out noise from first 60-70 samples of initial audio input
    for array_index in range(57):
        raw[array_index] = 0

    if plot_debug:
        # Plot base audio signal
        fig, axs = plt.subplots(2, 2, figsize=(9, 9))
        fig.tight_layout(pad=3.0)
        axs[0, 0].plot(np.linspace(0, len(raw)/44e3, len(raw)), raw)

        if type_of_scrambling == "scrambling":
            axs[0, 0].set_title('Raw audio signal')
        else:
            axs[0, 0].set_title('Scrambled audio signal')

        axs[0, 0].set_ylabel('Amplitude [a.u.]')

    # Calculate only positive frequency components of audio signal, so as to save computation time
    # and get real values directly from the inverse fourier transform
    raw_freq_amp = rfft(raw)
    raw_freq = rfftfreq(len(raw), 1/44e3)

    # Save original spectrum for debugging purposes
    initial_freq_amp = deepcopy(raw_freq_amp)

    if plot_debug:
        # Plot base frequency spectrum - Note 1/n factor to scale spectrum amplitude
        axs[0, 1].plot(raw_freq/1e3, 1/len(raw)*np.abs(raw_freq_amp))

        if type_of_scrambling == "scrambling":
            axs[0, 1].set_title('Raw signal spectrum')
        else:
            axs[0, 1].set_title('Scrambled signal spectrum')

        axs[0, 1].set_ylabel('Spectral amplitude [a.u.]')

    def shuffle_freqs(band1, band2, width, freqs, freq_amp):
        """Exchange frequency amplitude values for two given frequency bands with a given width.

        Args:
            band1(float): Center frequency in kHz of the first band
            band2 (float): Center frequency in kHz of the second band
            width (float): Width of bands in kHz
            freqs (list): List of all frequencies
            freq_amp(list): List of amplitudes for all frequencies

        Returns:
            freq_amp(list): Shuffled list of frequency amplitudes
        """

        # Scale to Hz
        band1 = band1*1e3
        band2 = band2*1e3
        width = width*1e3

        # Calculate number of data points in the positive frequency region
        freqs_num_of_data_points = len(freqs)

        def calc_band_boundaries(center_freq, freqs_num_of_data_points, width, freqs):
            """Given a center frequency in a spectrum, this method calculates the spectral value of the
            two band edges - The lower edge and the higher edge.

            Params:
            center_freq (int): Center frequency of band to be calculated, in kHz
            freqs_num_of_data_points (int): Number of data points in the positive spectral region
            width (int): Width of the band to be constructed, given in kHz
            freqs (list): List of spectral frequencies, in Hz

            Returns:
            (list): Frequency values for the lower and upper boundary of the constructed band
            """
            # Translate a given frequency range to a number of data points in the frequency list
            num_data_points_half_width = freqs_num_of_data_points * round(width*0.5) / freqs[-1]

            # Add/subtract the number of data points from the index of the center frequency to
            # create the band boundaries
            specific_width_high = center_freq + num_data_points_half_width
            specific_width_low = center_freq - num_data_points_half_width

            # Round the calculated index
            print([round(specific_width_low), round(specific_width_high)])
            return [round(specific_width_low), round(specific_width_high)]

        # BAND 1 CALCULATIONS #

        # Translate band frequency to index in frequency list
        band1_freq_index_pos = freqs_num_of_data_points * (band1 / freqs[-1])

        # Estimate the band boundaries for the given center band frequency
        band1_freqs_data_points_pos = calc_band_boundaries(band1_freq_index_pos, freqs_num_of_data_points, width, freqs)

        # BAND 2 CALCULATIONS #

        # Translate band frequency to index in frequency list
        band2_freq_index_pos = freqs_num_of_data_points * (band2 / freqs[-1])

        # Estimate the positive central band frequency boundaries for band 2
        band2_freqs_data_points_pos = calc_band_boundaries(band2_freq_index_pos, freqs_num_of_data_points, width, freqs)

        # Make sure that each window has the same sample length
        # This error will occur due to the discrete nature of digital sampling
        diff_band1 = band1_freqs_data_points_pos[1]-band1_freqs_data_points_pos[0]
        diff_band2 = band2_freqs_data_points_pos[1]-band2_freqs_data_points_pos[0]
        if diff_band1 > diff_band2:
            band2_freqs_data_points_pos[1] = band2_freqs_data_points_pos[1] + 1
            print('band_diff b1>b2')
        elif diff_band1 < diff_band2:
            band1_freqs_data_points_pos[1] = band1_freqs_data_points_pos[1] + 1
            print('band_diff b2>b1')

        # Copy the references to the frequenciy values in the given frequency bands
        band1_pos_vals = freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]].copy()
        band2_pos_vals = freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]].copy()

        # Insert the references into the original list of frequency amplitudes
        freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]] = band2_pos_vals
        freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]] = band1_pos_vals

        return freq_amp

    # Start timer for timing the duration of the shuffling
    t_start = time()

    # Number of time the shuffling operation is to be run
    n = len(key)

    # Start shuffling
    iterator = 1
    while iterator < n+1:
        # Access each entry of the key and shuffle the frequencies one step at a time
        band1_center_freq = key[iterator-1]['b1_cfreq']
        band2_center_freq = key[iterator-1]['b2_cfreq']

        # Swap the frequency values of the two bands with a given frequency width w
        raw_freq_amp = shuffle_freqs(band1_center_freq, band2_center_freq, w, raw_freq, raw_freq_amp)

        # Increase iterator
        print(iterator, end='\r')
        iterator += 1

    # Rename the shuffled frequency list for clarity
    shuffled_freq_amp = raw_freq_amp

    # Calculate the elapsed time for shuffling
    t_end = time()
    t_elapsed = t_end - t_start
    print("Time elapsed: " + str(t_elapsed) + " seconds")

    if plot_debug:
        # Plot the frequency spectrum after shuffling
        axs[1, 1].plot(raw_freq*1e-3, 1/len(raw)*np.abs(shuffled_freq_amp))
        axs[1, 1].set_xlabel("Frequency [kHz]")
        axs[1, 1].set_ylabel("Spectrum amplitude [a.u.]")

        if type_of_scrambling == "scrambling":
            axs[1, 1].set_title("Scrambled frequency spectrum")
        else:
            axs[1, 1].set_title("De-scrambled frequency spectrum")

    # Inverse FFT and convert the complex frequency values to real values
    scrambled_time_signal = irfft(shuffled_freq_amp, n=len(raw))

    if plot_debug:
        # Plot the shuffled signal in time
        axs[1, 0].plot(np.linspace(0, len(raw)/44e3, len(raw)), scrambled_time_signal)

        if type_of_scrambling == "scrambling":
            axs[1, 0].set_title('Scrambled time signal')
        else:
            axs[1, 0].set_title('De-scrambled time signal')

        axs[1, 0].set_xlabel('Time [s]')
        axs[1, 0].set_ylabel('Amplitude [a.u.]')

    # Convert the amplitude values to integers, such that they can be stored in a C-array
    scrambled_time_signal = [round(x) for x in scrambled_time_signal.tolist()]

    # Create scrambled audio file
    scrambled_audio_raw = array(audio_sample_size)
    scrambled_audio_raw.fromlist(scrambled_time_signal)

    # Save scrambled or de-scrambled audio file under an appropiate name depending on the type of operation
    if type_of_scrambling == 'scrambling':
        fn_end = 'Scrambled_audio.wav'
    else:
        fn_end = 'De_scrambled_audio.wav'

    with wave.open(fn_end, mode='wb') as scrambled_audio:
        scrambled_audio.setnchannels(nchan)
        scrambled_audio.setsampwidth(sample_width)
        scrambled_audio.setframerate(sample_freq)
        scrambled_audio.setnframes(len(raw))
        scrambled_audio.writeframes(scrambled_audio_raw)

    # Test return spectra for comparison
    return initial_freq_amp, shuffled_freq_amp

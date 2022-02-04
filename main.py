"""Main module"""
# Audio scramble as an encryption method


import wave
import os
from array import array
from time import time

import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt

# Import audio file
filename = os.path.join(os.getcwd(), 'Recording.wav')

# Check for the audio sample size, given in bytes
with wave.open(filename, mode='rb') as wav:
    nchan, sample_width, sample_freq, total_samples, _, _ = wav.getparams()

possible_sample_sizes = {1: 'B', 2: 'h', 4: 'i'}
audio_sample_size = possible_sample_sizes[sample_width]

# Create array with the same type as the audio sample size
raw = array(audio_sample_size)
raw.fromfile(open(filename, 'rb'), int(os.path.getsize(filename)/raw.itemsize))

# Clean out noise from first 60-70 samples
raw = raw[56:]

# Plot base audio signal
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
fig.tight_layout(pad=3.0)
axs[0, 0].plot(np.linspace(0, len(raw)/44e3, len(raw)), raw)
axs[0, 0].set_title('Raw audio signal')
axs[0, 0].set_ylabel('Amplitude [a.u.]')

# Calculate only positive frequency components of audio signal, so as to save computation time
# and get real values directly from the inverse fourier transform
raw_freq_amp = rfft(raw)
raw_freq = rfftfreq(len(raw), 1/44e3)

# Plot base frequency spectrum - Note 1/n factor to scale spectrum amplitude
axs[0, 1].plot(raw_freq/1e3, 1/len(raw)*np.abs(raw_freq_amp))
axs[0, 1].set_title('Raw signal spectrum')
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

        # Add/subtract the number of data points from the index of the center frequency to create the band boundaries
        specific_width_high = center_freq + num_data_points_half_width
        specific_width_low = center_freq - num_data_points_half_width

        # Round the calculated index
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
    elif diff_band1 < diff_band2:
        band1_freqs_data_points_pos[1] = band1_freqs_data_points_pos[1] + 1

    # Copy the references to the frequenciy values in the given frequency bands
    band1_pos_vals = freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]].copy()
    band2_pos_vals = freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]].copy()

    # Insert the references into the original list of frequency amplitudes
    freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]] = band2_pos_vals
    freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]] = band1_pos_vals

    return freq_amp


# According to https://en.wikipedia.org/wiki/Voice_frequency, the voice frequencies goes from around 0.3 to 3.4 kHz.
# We should therefore focus our shuffling in that area.
# Shuffle randomly in voice frequency area

def get_shuffle_freqs(w, max_freq):
    """Get pair of center frequencies to be shuffled with each other.

    A pseudo-random set of two frequency are selected as of now, but in the future
    this function will select the frequencies as described in an encryption key, which is
    a list of specific frequencies to be shuffled.

    Args:
        w (float): Spectral full width of shuffling frequency band
        max_freq (float): Upper limit of frequencies to be shuffled

    Returns:
        band1_freq, band2_freq (float): Center frequencies to be shuffled
    """
    # Select two pseudo-random center frequncies for shuffling
    band1_center_freq = np.random.uniform(low=w, high=max_freq)
    band2_center_freq = np.random.uniform(low=w, high=max_freq)

    # Check that the two selected bands are not too close to each other
    while abs(band1_center_freq-band2_center_freq) < w:
        band1_center_freq = np.random.uniform(low=w, high=max_freq)
        band2_center_freq = np.random.uniform(low=w, high=max_freq)
        print('Random numbers to close to each other')

    return band1_center_freq, band2_center_freq


# Start timer for timing the duration of the shuffling
t_start = time()

# Number of time shuffling operation is to be run
n = 40e3

# Band freqency width, given in kHz
w = 0.1

# Maximum frequency to be shuffled, given in kHz
max_freq = 15

# Start shuffling
iterator = 1
while iterator < n+1:

    # Generate random selection of center frequencies for the two bands
    band1_center_freq, band2_center_freq = get_shuffle_freqs(w, max_freq)

    # Swap the frequency values of the two bands with a given frequency width w
    raw_freq_amp = shuffle_freqs(band1_center_freq, band2_center_freq, w, raw_freq, raw_freq_amp)

    # Increase iterator
    print(iterator)
    iterator += 1

# Rename the shuffled frequency list for clarity
shuffled_freq_amp = raw_freq_amp

# Calculate the elapsed time for shuffling
t_end = time()
t_elapsed = t_end - t_start
print("Time elapsed: " + str(t_elapsed) + " seconds")

# Plot the frequency spectrum after shuffling
axs[1, 1].plot(raw_freq*1e-3, 1/len(raw)*np.abs(shuffled_freq_amp))
axs[1, 1].set_xlabel("Frequency [kHz]")
axs[1, 1].set_ylabel("Spectrum amplitude [a.u.]")
axs[1, 1].set_title("Scrambled frequency spectrum")

# Inverse FFT and convert the complex frequency values to real values
scrambled_time_signal = irfft(shuffled_freq_amp, n=len(raw))

# Plot the shuffled signal in time
axs[1, 0].plot(np.linspace(0, len(raw)/44e3, len(raw)), scrambled_time_signal)
axs[1, 0].set_title('Scrambled time signal')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Amplitude [a.u.]')

# Convert the amplitude values to integers, such that they can be stored in a C-array
scrambled_time_signal = [round(x) for x in scrambled_time_signal.tolist()]

# Create scrambled audio file
scrambled_audio_raw = array(audio_sample_size)
scrambled_audio_raw.fromlist(scrambled_time_signal)

with wave.open('Scramled_audio.wav', mode='wb') as scrambled_audio:
    scrambled_audio.setnchannels(nchan)
    scrambled_audio.setsampwidth(sample_width)
    scrambled_audio.setframerate(sample_freq)
    scrambled_audio.setnframes(len(raw))
    scrambled_audio.writeframes(scrambled_audio_raw)

plt.show()
# Descramble the audio using key

# Create descrambled audio file

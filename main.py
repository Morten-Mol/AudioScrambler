""""Main module"""
# Audio scramble as an encryption method

import wave
import os
from array import array
from time import time

import numpy as np
from scipy.fft import fft, fftfreq, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt


# Import audio file
filename = os.getcwd()+'/Recording.wav'

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
plt.figure()
plt.plot(np.linspace(0, len(raw)/44e3, len(raw)), raw)
plt.title('Raw audio signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [a.u.]')

# Generate key for filtering
raw_freq_amp = fft(raw)
raw_freq = fftfreq(len(raw), 1/44e3)[:len(raw)//2]

plt.figure()
plt.plot(raw_freq/1e3, 2/len(raw)*np.abs(raw_freq_amp[0:len(raw)//2]))
plt.title('Raw signal spectrum')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Spectral amplitude [a.u.]')

# Get symmetric representation of frequencies of raw signal
freqs = fftshift(fftfreq(len(raw), 1/44e3))
freq_amp = fftshift(raw_freq_amp)


def shuffle_freqs(band1, band2, width, freqs, freq_amp):
    """Exchange frequency amplitude values for two given frequency bands with a given width.

    Args:
        band1(float): Center frequency in kHz of the first band
        band2 (float): Center frequency in kHz of the second band
        width (float): Width of bands in kHz
        freqs (list): List of all frequencies
        freq_amp(list): List of amplitudes for all frequencies
    """

    # Scale to kHz
    band1 = band1*1e3
    band2 = band2*1e3
    width = width*1e3

    # Calculate number of data points in the positive frequency region
    freqs_num_of_data_points = len(freqs) - len(freqs)//2

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
        specific_width_high = center_freq + \
            freqs_num_of_data_points * round(width*0.5) / freqs[-1]
        specific_width_low = center_freq - \
            freqs_num_of_data_points * round(width*0.5) / freqs[-1]
        return [round(specific_width_low), round(specific_width_high)]

    # Estimate the central band frequency by interpolating the position using sc
    specific_freq = round(freqs_num_of_data_points +
                          freqs_num_of_data_points * (band1 / freqs[-1]))

    # Estimate the specific upper and lower frequency limits of the band
    band1_freqs_data_points_pos = calc_band_boundaries(
        specific_freq, freqs_num_of_data_points, width, freqs)

    specific_freq = round(freqs_num_of_data_points -
                          freqs_num_of_data_points * (band1 / freqs[-1]))
    specific_width_high = specific_freq + \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]
    specific_width_low = specific_freq - \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]

    band1_freqs_data_points_neg = [
        round(specific_width_low), round(specific_width_high)]

    specific_freq = round(freqs_num_of_data_points +
                          freqs_num_of_data_points * (band2 / freqs[-1]))
    specific_width_high = specific_freq + \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]
    specific_width_low = specific_freq - \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]

    band2_freqs_data_points_pos = [
        round(specific_width_low), round(specific_width_high)]

    specific_freq = round(freqs_num_of_data_points -
                          freqs_num_of_data_points * (band2 / freqs[-1]))
    specific_width_high = specific_freq + \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]
    specific_width_low = specific_freq - \
        freqs_num_of_data_points * round(width*0.5) / freqs[-1]

    band2_freqs_data_points_neg = [
        round(specific_width_low), round(specific_width_high)]

    # Make sure that each window has the same sample length
    # This error will occur due to the discrete nature of the frequency sampling
    diff_band1 = band1_freqs_data_points_pos[1]-band1_freqs_data_points_pos[0]
    diff_band2 = band2_freqs_data_points_pos[1]-band2_freqs_data_points_pos[0]
    if diff_band1 > diff_band2:
        band2_freqs_data_points_pos[1] = band2_freqs_data_points_pos[1] + 1
        band2_freqs_data_points_neg[1] = band2_freqs_data_points_neg[1] + 1
    elif diff_band1 < diff_band2:
        band1_freqs_data_points_pos[1] = band1_freqs_data_points_pos[1] + 1
        band1_freqs_data_points_neg[1] = band1_freqs_data_points_neg[1] + 1

    band1_pos_vals = freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]].copy()
    band1_neg_vals = freq_amp[band1_freqs_data_points_neg[0]:band1_freqs_data_points_neg[1]].copy()

    band2_pos_vals = freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]].copy()
    band2_neg_vals = freq_amp[band2_freqs_data_points_neg[0]:band2_freqs_data_points_neg[1]].copy()

    freq_amp[band1_freqs_data_points_pos[0]:band1_freqs_data_points_pos[1]] = band2_pos_vals
    freq_amp[band2_freqs_data_points_pos[0]:band2_freqs_data_points_pos[1]] = band1_pos_vals

    freq_amp[band1_freqs_data_points_neg[0]:band1_freqs_data_points_neg[1]] = band2_neg_vals
    freq_amp[band2_freqs_data_points_neg[0]:band2_freqs_data_points_neg[1]] = band1_neg_vals

    return freq_amp


# According to https://en.wikipedia.org/wiki/Voice_frequency, the voice frequencies goes from around 0.3 to 3.4 kHz.
# We should therefore focus our shuffling in that area.
# Shuffle randomly in voice frequency area
t_start = time()
for iterator in range(40000):
    w = 0.2
    max_freq = 7
    start = np.random.uniform(low=w/max_freq) * max_freq
    end = np.random.uniform(low=w/max_freq) * max_freq
    while abs(start-end) < w:
        end = np.random.uniform(low=w/max_freq) * max_freq
        print('Random numbers to close to each other')
    freq_amp = shuffle_freqs(start, end, w, freqs, freq_amp)
    print(iterator)
t_end = time()
t_elapsed = t_end - t_start
print("Time elapsed: " + str(t_elapsed) + " seconds")

raw_freq_amp = ifftshift(freq_amp)

scrambled_signal_freq_amp = raw_freq_amp
fig, ax = plt.subplots()
ax.plot(raw_freq, 2/len(raw)*np.abs(scrambled_signal_freq_amp[0:len(raw)//2]))
ax.set_xlabel("Frequency [kHz]", fontsize=14)
ax.set_ylabel("Spectrum amplitude [a.u.]", color="blue", fontsize=14)

plt.figure()
plt.plot(fftshift(scrambled_signal_freq_amp))

scrambled_signal = ifft(scrambled_signal_freq_amp)
scrambled_signal = scrambled_signal.real

plt.figure()
plt.plot(scrambled_signal)
plt.title('Scrambled signal')


scrambled_signal = [round(x) for x in scrambled_signal.tolist()]

# Create scrambled audio file
scrambled_audio_raw = array(audio_sample_size)
scrambled_audio_raw.fromlist(scrambled_signal)

with wave.open('Scramled_audio.wav', mode='wb') as scrambled_audio:
    scrambled_audio.setnchannels(nchan)
    scrambled_audio.setsampwidth(sample_width)
    scrambled_audio.setframerate(sample_freq)
    scrambled_audio.setnframes(len(raw))
    scrambled_audio.writeframes(scrambled_audio_raw)

plt.show()
# Descramble the audio using key

# Create descrambled audio file

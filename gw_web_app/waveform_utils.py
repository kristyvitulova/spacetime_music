# waveform_utils.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from pycbc.waveform import get_td_waveform
import wave
import os

def generate_gw_waveform(mass1, mass2, spin1z, spin2z,
                         delta_t=1.0 / 4096, f_lower=20.0):
    mass1, mass2, spin1z, spin2z = validate_parameters(
        mass1, mass2, spin1z, spin2z)
    approximant = 'IMRPhenomD'
    try:
        hp, _ = get_td_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            delta_t=delta_t,
            f_lower=f_lower
        )
    except Exception as e:
        raise RuntimeError(f"Error generating waveform: {e}")
    waveform = hp.data
    sample_rate = 1.0 / delta_t
    return waveform, sample_rate

def validate_parameters(mass1, mass2, spin1z, spin2z):
    mass1 = max(float(mass1), 1.0)
    mass2 = max(float(mass2), 1.0)
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
    spin1z = max(min(float(spin1z), 1.0), -1.0)
    spin2z = max(min(float(spin2z), 1.0), -1.0)
    return mass1, mass2, spin1z, spin2z

def resample_waveform(waveform, original_sample_rate, target_sample_rate):
    duration = len(waveform) / original_sample_rate
    num_samples = int(duration * target_sample_rate)
    resampled_waveform = resample(waveform, num_samples)
    return resampled_waveform

def normalize_waveform(waveform):
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

def frequency_shift(waveform, speed_factor):
    num_samples = int(len(waveform) / speed_factor)
    if num_samples < 1:
        num_samples = 1
    shifted_waveform = resample(waveform, num_samples)
    return shifted_waveform

def save_waveform_to_wav(waveform, sample_rate, filename):
    waveform_int16 = np.int16(waveform * 32767)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(waveform_int16.tobytes())

def plot_waveform_with_specgram(waveform, sample_rate, title, filename):
    time_array = np.arange(len(waveform)) / sample_rate

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time_array, waveform)
    axs[0].set_title(f"Gravitational Waveform: {title}")
    axs[0].set_ylabel("Strain")
    axs[0].grid(True)

    window_duration = 0.05
    nperseg = int(sample_rate * window_duration)
    noverlap = int(nperseg * 0.5)

    f, t, Sxx = spectrogram(
        waveform, fs=sample_rate,
        window='hann', nperseg=nperseg, noverlap=noverlap
    )

    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    im = axs[1].pcolormesh(
        t, f, Sxx_log, shading='gouraud', cmap='inferno'
    )
    axs[1].set_yscale('log')
    axs[1].set_title('Spectrogram')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_ylim([20, 1500])
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].set_xlim([time_array[0], time_array[-1]])

    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label('Power/Frequency (dB/Hz)')

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

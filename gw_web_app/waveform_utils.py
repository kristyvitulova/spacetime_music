# waveform_utils.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from pycbc.waveform import get_td_waveform
import wave
import os

def generate_gw_waveform(mass1, mass2,
                         spin1x, spin1y, spin1z,
                         spin2x, spin2y, spin2z,
                         delta_t=1.0 / 4096, f_lower=20.0,
                         approximant='IMRPhenomPv2'):
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = validate_parameters(
        mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
    try:
        hp, _ = get_td_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=spin1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=spin2z,
            delta_t=delta_t,
            f_lower=f_lower
        )
    except Exception as e:
        raise RuntimeError(f"Error generating waveform: {e}")
    waveform = hp.data
    sample_rate = 1.0 / delta_t
    return waveform, sample_rate

def validate_parameters(mass1, mass2,
                        spin1x, spin1y, spin1z,
                        spin2x, spin2y, spin2z):
    mass1 = max(float(mass1), 1.0)
    mass2 = max(float(mass2), 1.0)
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1x, spin2x = spin2x, spin1x
        spin1y, spin2y = spin2y, spin1y
        spin1z, spin2z = spin2z, spin1z
    spin1x = max(min(float(spin1x), 1.0), -1.0)
    spin1y = max(min(float(spin1y), 1.0), -1.0)
    spin1z = max(min(float(spin1z), 1.0), -1.0)
    spin2x = max(min(float(spin2x), 1.0), -1.0)
    spin2y = max(min(float(spin2y), 1.0), -1.0)
    spin2z = max(min(float(spin2z), 1.0), -1.0)
    return mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z

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
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Ensure waveform uses float32 data type
    waveform = waveform.astype(np.float32)

    # Original time array
    t_original = np.arange(len(waveform)) / sample_rate

    # Find the time of peak amplitude (merger time)
    idx_peak = np.argmax(np.abs(waveform))
    t_merger = t_original[idx_peak]

    # Compute time before merger (positive values before merger)
    t_before_merger = t_merger - t_original

    # Define time range: from t_max down to t_min seconds before merger
    t_min = 1e-4    # 0.0001 seconds (0.1 ms)
    t_max = 100.0   # 100 seconds

    # Create mask for desired time range
    mask = (t_before_merger >= t_min) & (t_before_merger <= t_max)

    # Apply mask to time and waveform arrays
    t_plot = t_before_merger[mask]
    waveform_plot = waveform[mask]

    # Reverse the arrays so that time increases from left to right
    t_plot = t_plot[::-1]
    waveform_plot = waveform_plot[::-1]

    # Set up the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the waveform
    axs[0].plot(t_plot, waveform_plot)
    axs[0].set_title(f"Gravitational Waveform: {title}")
    axs[0].set_ylabel("Strain")

    # Set logarithmic scale for x-axis
    axs[0].set_xscale('log')
    axs[0].set_xlim(t_min, t_max)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Configure x-axis ticks
    tick_values = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    axs[0].set_xticks(tick_values)
    axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axs[0].set_xlabel('Time before Merger (s)')

    # Compute spectrogram for the same time range
    waveform_spec = waveform_plot
    nperseg = int(sample_rate * 0.002)  # 2 ms window
    noverlap = int(nperseg * 0.9)       # 90% overlap

    f, t_spec, Sxx = spectrogram(
        waveform_spec,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )

    # Adjust spectrogram time array to align with t_plot
    # Since waveform_spec is reversed, t_spec needs to be adjusted
    t_spec_plot = t_plot[0] + t_spec

    # Plot spectrogram
    extent = [t_spec_plot[0], t_spec_plot[-1], f[0], f[-1]]
    im = axs[1].imshow(
        10 * np.log10(Sxx + 1e-10),
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='inferno'
    )

    axs[1].set_title('Spectrogram')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_xscale('log')
    axs[1].set_xlim(t_min, t_max)
    axs[1].set_ylim([20, 1500])  # Adjust frequency limits as needed
    axs[1].set_yscale('log')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Configure x-axis ticks for the spectrogram
    axs[1].set_xticks(tick_values)
    axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axs[1].set_xlabel('Time before Merger (s)')

    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label('Power/Frequency (dB/Hz)')

    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)

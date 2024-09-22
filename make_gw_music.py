# gw_waveform_save.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from pycbc.waveform import get_td_waveform
import wave
from tqdm import tqdm
import timeit


def generate_gw_waveform(mass1, mass2, spin1z, spin2z,
                         delta_t=1.0 / 4096, f_lower=20.0):
    mass1, mass2, spin1z, spin2z = validate_parameters(
        mass1, mass2, spin1z, spin2z)
    approximant = 'IMRPhenomD'
    try:
        hp, _ = get_td_waveform(approximant=approximant,
                                mass1=mass1,
                                mass2=mass2,
                                spin1z=spin1z,
                                spin2z=spin2z,
                                delta_t=delta_t,
                                f_lower=f_lower)
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


def save_waveform_data(waveform, filename):
    np.savez_compressed(filename, waveform=waveform)


def main():
    start_time = timeit.default_timer()

    if not os.path.exists('Data'):
        os.makedirs('Data')
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    test_cases = [
        {
            'mass1': 30, 'mass2': 30, 'spin1z': 0.0,
            'spin2z': 0.0, 'label': 'Equal_Mass_No_Spin'
        },
        {
            'mass1': 35, 'mass2': 25, 'spin1z': 0.5,
            'spin2z': -0.5, 'label': 'Unequal_Mass_Opposite_Spins'
        },
        {
            'mass1': 20, 'mass2': 10, 'spin1z': 0.9,
            'spin2z': 0.9, 'label': 'High_Spins_Unequal_Mass'
        },
        {
            'mass1': 50, 'mass2': 50, 'spin1z': -0.9,
            'spin2z': -0.9, 'label': 'High_Negative_Spins_Equal_Mass'
        },
    ]

    for case in tqdm(test_cases, desc="Processing waveforms"):
        mass1 = case['mass1']
        mass2 = case['mass2']
        spin1z = case['spin1z']
        spin2z = case['spin2z']
        label = case['label']

        try:
            waveform, sample_rate = generate_gw_waveform(
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z
            )
            speed_factor = 1.0
            waveform = frequency_shift(waveform, speed_factor)
            target_sample_rate = 44100
            waveform = resample_waveform(
                waveform, sample_rate, target_sample_rate)
            waveform = normalize_waveform(waveform)

            filename_base = (f"waveform_m1_{mass1}_m2_{mass2}_"
                             f"s1z_{spin1z}_s2z_{spin2z}")

            plot_filename = os.path.join('Figures', filename_base + ".png")
            plot_waveform(waveform, target_sample_rate,
                          label, plot_filename)
            wav_filename = os.path.join('Data', filename_base + ".wav")
            save_waveform_to_wav(waveform, target_sample_rate, wav_filename)
            waveform_data_filename = os.path.join(
                'Data', filename_base + ".npz")
            save_waveform_data(waveform, waveform_data_filename)
        except RuntimeError as e:
            print(f"Error processing {label}: {e}")

    elapsed = timeit.default_timer() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")


def plot_waveform(waveform, sample_rate, title, filename):
    time_array = np.arange(len(waveform)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_array, waveform)
    ax.set_title(f"Gravitational Waveform: {title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    main()

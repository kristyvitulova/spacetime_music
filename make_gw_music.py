# gw_waveform_save.py

import os
import timeit
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample, spectrogram
from pycbc.waveform import get_td_waveform
import wave
from tqdm import tqdm


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


def save_waveform_data(waveform, filename):
    np.savez_compressed(filename, waveform=waveform)


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate gravitational waveforms.'
    )
    parser.add_argument(
        '--params_file', type=str, default=None,
        help='Path to a YAML file containing list of parameter sets'
    )
    parser.add_argument('--mass1', type=float, default=30.0,
                        help='Mass of the first black hole (solar masses)')
    parser.add_argument('--mass2', type=float, default=30.0,
                        help='Mass of the second black hole (solar masses)')
    parser.add_argument('--spin1z', type=float, default=0.0,
                        help='Spin of the first black hole (-1 to 1)')
    parser.add_argument('--spin2z', type=float, default=0.0,
                        help='Spin of the second black hole (-1 to 1)')
    parser.add_argument('--delta_t', type=float, default=1.0 / 4096,
                        help='Time step between samples')
    parser.add_argument('--f_lower', type=float, default=20.0,
                        help='Lower frequency cutoff (Hz)')
    parser.add_argument('--speed_factor', type=float, default=1.0,
                        help='Factor to speed up or slow down the waveform')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Target sample rate for audio (Hz)')
    parser.add_argument('--label', type=str, default='GW_Waveform',
                        help='Label for the output files')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output files')
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = timeit.default_timer()

    data_dir = os.path.join(args.output_dir, 'Data')
    figures_dir = os.path.join(args.output_dir, 'Figures')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if args.params_file:
        with open(args.params_file, 'r') as f:
            test_cases = yaml.safe_load(f)
    else:
        test_cases = [{
            'mass1': args.mass1,
            'mass2': args.mass2,
            'spin1z': args.spin1z,
            'spin2z': args.spin2z,
            'label': args.label,
            'delta_t': args.delta_t,
            'f_lower': args.f_lower,
            'speed_factor': args.speed_factor,
            'sample_rate': args.sample_rate
        }]

    processing_times = []

    for case in tqdm(test_cases, desc="Processing waveforms"):
        mass1 = case.get('mass1', 30.0)
        mass2 = case.get('mass2', 30.0)
        spin1z = case.get('spin1z', 0.0)
        spin2z = case.get('spin2z', 0.0)
        label = case.get('label', 'GW_Waveform')
        delta_t = case.get('delta_t', 1.0 / 4096)
        f_lower = case.get('f_lower', 20.0)
        speed_factor = case.get('speed_factor', 1.0)
        sample_rate = case.get('sample_rate', 44100)

        try:
            processing_start = timeit.default_timer()

            waveform, original_sample_rate = generate_gw_waveform(
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                delta_t=delta_t,
                f_lower=f_lower
            )
            waveform = frequency_shift(waveform, speed_factor)
            waveform = resample_waveform(
                waveform, original_sample_rate, sample_rate)
            waveform = normalize_waveform(waveform)

            processing_end = timeit.default_timer()
            processing_time = processing_end - processing_start
            processing_times.append(processing_time)

            filename_base = (
                f"{label}_m1_{mass1}_m2_{mass2}_"
                f"s1z_{spin1z}_s2z_{spin2z}"
            )

            waveform_data_filename = os.path.join(
                data_dir, filename_base + ".npz"
            )
            save_waveform_data(waveform, waveform_data_filename)

            wav_filename = os.path.join(
                data_dir, filename_base + ".wav"
            )
            save_waveform_to_wav(waveform, sample_rate, wav_filename)

        except RuntimeError as e:
            print(f"Error processing {label}: {e}")
            continue

    for case in test_cases:
        mass1 = case.get('mass1', 30.0)
        mass2 = case.get('mass2', 30.0)
        spin1z = case.get('spin1z', 0.0)
        spin2z = case.get('spin2z', 0.0)
        label = case.get('label', 'GW_Waveform')
        sample_rate = case.get('sample_rate', 44100)

        filename_base = (
            f"{label}_m1_{mass1}_m2_{mass2}_"
            f"s1z_{spin1z}_s2z_{spin2z}"
        )

        waveform_data_filename = os.path.join(
            data_dir, filename_base + ".npz"
        )
        data = np.load(waveform_data_filename)
        waveform = data['waveform']

        plot_filename = os.path.join(
            figures_dir, filename_base + ".png"
        )
        plot_waveform_with_specgram(
            waveform, sample_rate, label, plot_filename
        )

    elapsed = timeit.default_timer() - start_time
    total_processing_time = sum(processing_times)
    print(
        f"Total processing time (excluding plotting): "
        f"{total_processing_time:.2f} seconds"
    )
    print(f"Total elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

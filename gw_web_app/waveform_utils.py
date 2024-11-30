# waveform_utils.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.signal import spectrogram
import wave
import gwsurrogate

def generate_gw_waveform(mass1, mass2,
                         spin1x, spin1y, spin1z,
                         spin2x, spin2y, spin2z,
                         delta_t=1.0 / 4096):
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = validate_parameters(
        mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
    try:
        sur = gwsurrogate.LoadSurrogate('NRSur7dq4')
        q = mass1/mass2
        chiA = [spin1x, spin1y, spin1z]
        chiB = [spin2x, spin2y, spin2z]
        dt = delta_t
        M = mass1 + mass2
        
        # Lets just assume a distance of 100 Mpc for the sake of having seconds as the time axis
        dist_mpc = 100
        domain, h, _ = sur(q, chiA, chiB, dt=dt, f_low=0, M=M, dist_mpc=dist_mpc, units='mks')
        sample_rate = 1.0 / dt
        return domain, h, sample_rate
    except Exception as e:
        print("Error generating waveform: ", e)

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


def normalize_waveform(waveform):
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def save_waveform_to_wav(waveform, sample_rate, filename):
    waveform_int16 = np.int16(waveform * 32767)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(waveform_int16.tobytes())


def plot_ng_waveform_with_specgram(domain, waveform, sample_rate, title, filename):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    t_max = max(domain)
    t_min = -0.05
    
    # Set up the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))


    # Plot the waveform
    #axs[0].plot(domain, h22, label='l2m2 real')
    axs[0].plot(domain, waveform)
    axs[0].set_title(f"Gravitational Waveform: {title}")
    axs[0].set_ylabel('Strain', fontsize=18)
    axs[0].set_xlabel('Time before Merger (s)')
    axs[0].set_xlim(t_min, t_max)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # Compute spectrogram for the same time range
    waveform_spec = waveform
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
    t_spec_plot = domain[0] + t_spec

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
    axs[1].set_xlim(t_min, t_max)
    axs[1].set_ylim([20, 1500])  # Adjust frequency limits as needed
    axs[1].set_yscale('log')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Configure x-axis ticks for the spectrogram
    axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axs[1].set_xlabel('Time before Merger (s)')

    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label('Power/Frequency (dB/Hz)')


    fig.tight_layout()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)



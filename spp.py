import numpy as np
from scipy.fft import fft, ifft
from scipy.special import expi

def psdsmooth(sig2, alpha):
    # Initialize variables
    win_len = alpha * 2 + 1
    win = np.ones(win_len) / win_len
    # Apply the filter
    return np.convolve(sig2, win, mode='same')

def compute_spp(x, fs):
    # Read the noisy speech file
    frame_len = 256
    step_len = 128
    # Initialize variables
    win = np.hanning(frame_len)  # define window
    sys = 0.5 / (win[:step_len] + win[step_len:frame_len])

    # Noise magnitude calculations - assuming that the first 6 frames is noise/silence
    nfft = 2 * frame_len

    # Allocate memory and initialize various variables
    x_old = np.zeros(step_len)
    nframes = int((len(x)-frame_len) // step_len) + 1
    xfinal = np.zeros(nframes * step_len)
    p = np.zeros(nfft)
    pest = np.zeros(nfft)
    tkm = np.ones(nfft) * 5 # 5dB in terms of SNR
    
    # for output
    noisy_psd = np.zeros((nfft, nframes))
    est_noise = np.zeros((nfft, nframes))

    # Start Processing
    k = 0
    aa = 0.98
    alphas = 0.8
    alpha = .95
    alphap = .2
    ksi_min = 10 ** (-25 / 10)
    D = int(fs * 0.8 / step_len)
    for n in range(nframes):
        insign = win * x[k:k + frame_len]
        spec = fft(insign, nfft)
        sig = np.abs(spec)  # compute the magnitude
        
        sig2 = sig ** 2
        if n == 0:
            noise_mu2 = sig2

        gammak = np.minimum(sig2 / noise_mu2, 40)  # limit post SNR to avoid overflows
        if n == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)  # a priori SNR
            ksi = np.maximum(ksi_min, ksi)  # limit ksi to -25 dB
        
        sf = psdsmooth(sig2, 1)
        # MCRA
        if n == 0:
            ss = sf
            smin = ss
            stmp = ss
        else:
            ss = alphas * ss + (1 - alphas) * sf
            
        if n % D == 0:
            smin = np.minimum(stmp, smin)
            stmp = ss
        else:
            smin = np.minimum(smin, ss)
            stmp = np.minimum(stmp, ss)
        sr = ss / smin
        p[sr>=tkm] = 1
        p[sr<tkm] = 0
        pest = alphap * pest + (1 - alphap) * p
        alphad = alpha + (1-alpha) * pest
        
        # log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        noise_mu2 = alphad * noise_mu2 + (1 - alphad) * sig2

        A = ksi / (1.0 + ksi)  # MMSE estimator
        vk = A * gammak
        ei_vk = 0.5 * expi(1, vk)
        hw = A * np.exp(ei_vk)

        sig = sig * hw
        xk_prev = sig ** 2

        xi_w = np.real(ifft(hw * spec, nfft))
        xfinal[k:k + step_len] = sys * (x_old + xi_w[:step_len])
        x_old = xi_w[step_len:frame_len]

        noisy_psd[:, n] = sig2
        est_noise[:, n] = noise_mu2
        k += step_len

    return xfinal, est_noise, noisy_psd

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import soundfile as sf
    x, srate = sf.read('data/mic.wav')  # nsdata is a column vector

    y, npsd, opsd = compute_spp(x, srate)
    sf.write('data/mic_spp.wav', y, srate)
    fig,ax = plt.subplots(3, 1)
    ax[0].pcolormesh(10*np.log10(opsd[0:257,:]), shading='gouraud', vmin=-80, vmax=0, cmap='jet')
    ax[0].set_title('Original PSD')
    fig.colorbar(ax[0].collections[0], ax=ax[0], orientation='vertical')
    ax[1].pcolormesh(10*np.log10(npsd[0:257,:]), shading='gouraud', vmin=-80, vmax=0, cmap='jet')
    ax[1].set_title('Noisy PSD')
    fig.colorbar(ax[1].collections[0], ax=ax[1], orientation='vertical')
    # return SNR in dB per frame
    snr = 10*np.log10(np.sum(opsd, axis=0)/np.sum(npsd, axis=0))
    ax[2].plot(snr)
    ax[2].set_title('SNR')
    ax[2].set_ylabel('dB')
    ax[2].set_xlim([0, len(snr)])
    ax[2].grid()
    plt.savefig('data/spp.png')

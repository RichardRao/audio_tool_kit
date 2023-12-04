import numpy as np
# a class for snr estimation in realtime

class SnrEstimation(object):
    def __init__(self, frame_len, step_len, fs):
        self.frameLen = int(frame_len)
        self.stepLen = int(step_len)
        self.nfft = int(frame_len)
        self.nfbins = int(self.nfft // 2 + 1)
        self.totalNum = int(4 * fs // step_len)
        self.firstRun = True
        self.decayFactor = .95
        self.skewThreshold = 6.0
        self.startFreqIdx = 1
        self.cutoffFreqIdx = int(4000 * self.nfft // fs)
        self.circularIndex = 0
        self.DB_NOISEFLOOR = -140
        self.HISTOGRAM_THRESHOLD = self.totalNum * 0.05
        self.psdSumVectorDb = np.ones(self.nfbins) * self.DB_NOISEFLOOR * self.totalNum
        self.psdDbCounter = np.ones((self.totalNum, self.nfbins)) * self.DB_NOISEFLOOR
        self.noisePsdInDb = np.zeros(self.nfbins)
        self.smoothedSnr = 0.0
        self.snr = 0.0
        
    def calculateLogSpectrum(self, psd):
        psd = psd[0:self.nfbins]
        #print(np.shape(psd))
        psd[psd==0] = 10**(self.DB_NOISEFLOOR/10)
        self.psdInDb = np.array(10.0*np.log10(psd)).astype(int)
        self.psdSumVectorDb -= self.psdDbCounter[self.circularIndex,:].flatten()
        self.psdDbCounter[self.circularIndex,:] = self.psdInDb
        self.psdSumVectorDb += self.psdDbCounter[self.circularIndex,:].flatten()
        self.circularIndex += 1
        self.circularIndex = 0 if self.circularIndex >= self.totalNum else self.circularIndex
        self.firstRun = False if self.firstRun and self.circularIndex == 0 else self.firstRun
        
    def calculateEstimateNoiseLevel(self):
        scaler = 1.0 / self.totalNum
        dB_step = 2
        dB_lvls = int(-(self.DB_NOISEFLOOR - 1) / dB_step) + 1
        # estimatedNoiseLevelMean = np.zeros(self.cutoffFreqIdx)
        estimatedNoiseLevelMean = self.psdSumVectorDb * scaler
        estimatedNoiseLevelMode = np.zeros(self.nfbins)
        maxCnt = np.zeros(self.nfbins)
        
        for i in range(self.nfbins):
            db_counts = np.zeros(dB_lvls)
            vals,counts = np.unique(self.psdDbCounter[:,i], return_counts=True)
            # print(self.psdDbCounter[:,i])
            for j,val in enumerate(vals):
                idx = int(-vals[j]//dB_step)
                db_counts[idx] += counts[j] 
            index = np.argmax(db_counts)
            estimatedNoiseLevelMode[i] = -index * dB_step - dB_step / 2
            maxCnt = db_counts[index]
        cond_index = \
            (np.abs(estimatedNoiseLevelMean-estimatedNoiseLevelMode) < self.skewThreshold) & \
            (maxCnt > self.HISTOGRAM_THRESHOLD)
        self.noisePsdInDb[cond_index] = estimatedNoiseLevelMean[cond_index]
        
    def calculateRawSnr(self):
        sums = np.mean(10.0**((self.psdInDb[self.startFreqIdx:self.cutoffFreqIdx] - self.noisePsdInDb[self.startFreqIdx:self.cutoffFreqIdx])*0.1))
        self.snr = 10.0 * np.log10(sums)
    
    def snrSmoothing(self):
        self.smoothedSnr = self.snr if self.snr > self.smoothedSnr else self.smoothedSnr*self.decayFactor
    
    ''' function to get SNR for a given frame
        input: psd 
    '''
    def getSnr(self, psd):
        self.calculateLogSpectrum(psd)
        
        self.calculateEstimateNoiseLevel()
        
        if self.firstRun is True:
            return np.nan
        
        self.calculateRawSnr()
            
        # Smoothing
        self.snrSmoothing()
        
        return self.smoothedSnr
    
if __name__ == "__main__":
    import soundfile as sf
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.fft import fft
    root_path = "/Users/richard/Documents/workspace/snr_estimation"
    x, fs = sf.read(os.path.join(root_path,"babble_noisy_speech.wav"))
    x = np.concatenate([x, x, x, x, x])
    frame_dur = 0.032
    frame_len = int(frame_dur * fs)
    step_len = frame_len // 2
    nframe = int((len(x) - frame_len) // step_len + 1)
    snrInst = SnrEstimation(frame_len, step_len, fs)
    # Get frames
    frames = np.array([x[i*step_len:i*step_len+frame_len] for i in range(nframe)])
    nfft = frame_len
    # Get PSDs
    specs = np.array([fft(frames[i, :], nfft) for i in range(nframe)])
    psds = np.abs(specs/nfft)**2
    # Get SNRs
    snrs = np.array([snrInst.getSnr(psds[fid,:].flatten()) for fid in range(nframe)])
    f,ax = plt.subplots(2,1)
    ax[0].plot(x)
    ax[1].plot(np.arange(0,nframe), snrs)
    plt.savefig(os.path.join(root_path, "snr.png"))
    

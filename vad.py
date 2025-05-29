'''
Filename: /home/richard/workspace/audio_tool_kit/vad.py
Path: /home/richard/workspace/audio_tool_kit
Created Date: Sunday, June 23rd 2024, 1:11:04 pm
Author: richard

Copyright (c) 2024
'''
import numpy as np
class vad():
    def __init__(self, signal, fs, frameDur, nbits, noise_threshold=-50):
        self._signal_len = len(signal)
        if len(np.shape(signal)) != 1:
            signal = signal[:,0].flatten()
        self._frame_size = int(frame_dur * fs)
        self._fs = fs
        
        step = .5 # measured in dB
        eps = 2**(-(nbits-1))
        signal[signal < eps] = eps
        self._nframe = len(signal) // self._frame_size
        self._ref_rms = np.zeros(size(signal))
        
        s = np.reshape(signal[0:self._nframe*self._frame_size], (-1, self._frame_size))
        self._ref_rms = 10*np.log10(np.mean(s**2, axis=1))
        self._smoothed_ref_rms = self._ref_rms
        
        for frameIndex in range(1, self._nframe):
            if self._smoothed_ref_rms[frameIndex] <= self._smoothed_ref_rms[frameIndex-1]:
                self._smoothed_ref_rms[frameIndex] = self._smoothed_ref_rms[frameIndex-1] - step
        self.activity_mask = self._smoothed_ref_rms >= noise_threshold
        self.noise_mask = self._smoothed_ref_rms < noise_threshold
    
    def get_active_mask(self):
        return self.activity_mask.flatten()
        
        
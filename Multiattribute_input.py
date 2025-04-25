#Multiattribute_input
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import remez, minimum_phase
#print(scipy.__version__)

# Loading SEAM model --target domain
model= np.load('Lowfreq_segyfiles/Seam_model_full.npy')[:,::2][:, 50:]
print(model.shape)

# generate ref series 
rc = np.zeros((1502,701))

for j in range(model.shape[1]-1):
    rc[:,j] = (model[:,j+1]-model[:,j])/(model[:,j]+model[:,j+1])
print(rc.shape)

#generate mini phase wavelet
freq = [0,10,60, 100]
fs = 200   # time sample=0.005ms
desired = [1,0]
h_linear = remez(2**5, freq, desired, fs=fs)
h_min= minimum_phase(h_linear, method='hilbert')

# visualization of min phase wavelet
plt.plot(h_min, c='C1')
plt.xlim(0,len(h_linear)-1)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(c='k', alpha=0.15)

# create syn seismic
syn_s = []
for i in range(rc.shape[0]):
    a = np.convolve(h_min, rc[i], mode='same')
    syn_s.append(a)
syn_s = np.array(syn_s, dtype='float32') 
print(syn_s.shape)


# syn seismic visulization
plt.figure(figsize=(3,2), dpi=300)
plt.imshow(syn_s.T,cmap='gray', vmin=-0.1, vmax=0.1)
plt.xlabel('traces')
plt.ylabel('time samples (ms)')
plt.colorbar()
plt.axis('tight')
plt.show();

# Bandlimit the generated syn seismic
# 1. Frequency filter
sos = signal.butter(5, [40,280], 'bandpass', fs=1000, output='sos')
syn_band = signal.sosfilt(sos, syn_s)
syn_band = syn_band.reshape((1,1502,701))

dt=4
nt_wav = 21
nfft = 2**11

t_wav = np.arange(nt_wav)*(dt/1000)
t_wav = np.concatenate((np.flipud(-t_wav[1:]),t_wav), axis=0)

#estimate wavelet spectrum
wav_est_fft = np.mean(np.abs(np.fft.fft(syn_band[...,:701], nfft, axis=-1)), axis=(0,1))
fwest = np.fft.fftfreq(nfft, d = dt/1000)

#create wavelet in time
wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
wav_est = wav_est/wav_est.max()
wcentre = np.argmax(np.abs(wav_est))

figure(figsize=(4,2), dpi=100)
plt.plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'm')
plt.vlines(x=5, ymin=0, ymax=0.4, color='red', zorder=1)
plt.xticks(np.arange(0,125, step=5))
plt.tight_layout()
plt.show()

# Generate seismic envelope and inst phase attribute using bandlimited seismic 
#1. Envelope
syn_band = syn_band.squeeze()
hil = hilbert(syn_band)
env = np.abs(hil)
print(env.shape)

#2. inst. phase
phase  = np.unwrap(np.angle(hil))
print(phase.shape)

# stacking genereted attributes
stack = np.dstack((syn_band, env, phase))
print(stack.shape)

input = stack.transpose((0,2,1))
#np.save('filename.npy', input, allow_pickle=True)




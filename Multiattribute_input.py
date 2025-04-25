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
model= np.load('data/Seam_model_full.npy')[:,::2][:, 50:]
print(model.shape)

# generate ref series 
rc = np.zeros((1502,701))

for j in range(model.shape[1]-1):
    rc[:,j] = (model[:,j+1]-model[:,j])/(model[:,j]+model[:,j+1])
print(rc.shape)

#create a minimum-phase wavelet
freq = [0,10,60, 100]
fs = 200   # time sample=0.005ms
desired = [1,0]
h_linear = remez(2**5, freq, desired, fs=fs)
h_min= minimum_phase(h_linear, method='hilbert')

# Visualization of minimum-phase wavelet
plt.plot(h_min, c='C1')
plt.xlim(0,len(h_linear)-1)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(c='k', alpha=0.15)

# create synthetic seismic
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

dt=5
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

#3. stacking generated attributes --Multi-attribute input
stack = np.dstack((syn_band, env, phase))
print(stack.shape)

input = stack.transpose((0,2,1))
#np.save('filename.npy', input, allow_pickle=True)

# Analysis of recovered missing low-frequency information using the envelope
syn_band = syn_band[np.newaxis,:,:]
env = env[np.newaxis,:,:]
# syn and env spectrum
dt=5
nt_wav=21
nfft=2**11

t_wav = np.arange(nt_wav)*(dt/1000)
t_wav = np.concatenate((np.flipud(-t_wav[1:]),t_wav), axis=0)

#estimate spectrum---synthetic seismic
wav_est_fft = np.mean(np.abs(np.fft.fft(syn_band[...,:701], nfft, axis=-1)), axis=(0,1))
fwest = np.fft.fftfreq(nfft, d = dt/1000)

#estimate spectrum---envelope
wav_est_fft_env = np.mean(np.abs(np.fft.fft(env[...,:701], nfft, axis=-1)), axis=(0,1))
fwest_env = np.fft.fftfreq(nfft, d = dt/1000)
#create wavelet in time
wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
wav_est = wav_est/wav_est.max()
wcentre = np.argmax(np.abs(wav_est))

#to smooth the envelope curve
from scipy.signal import savgol_filter
yhat= savgol_filter(wav_est_fft_env, 51,2) # window size 51, polynomial order 2

#function to normalize the data 
def normalizedata(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

figure(figsize=(8,3), dpi=300)
plt.plot(fwest[:nfft//2], normalizedata(wav_est_fft[:nfft//2]), 'b', label='Bandlimited seismic', lw=1.5)
plt.plot(fwest_env[:nfft//2], normalizedata(yhat[:nfft//2]), 'r', linestyle='--', label='Seismic envelope', lw=1.5)
plt.xticks(np.arange(0, 121, step=5))
plt.ylim(0,1)
plt.legend(loc='upper right')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Normalized amplitude')
plt.tight_layout()
plt.show()

# adding Gaussian noise as a data augmentation technique
#1. to seismic data
syn_band = syn_band.squeeze()
mu=0.0
std5=0.05*np.std(syn_band)
std10=0.1*np.std(syn_band)
std15=0.15*np.std(syn_band)
std20=0.2*np.std(syn_band)
def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu,std,size=x.shape)
    x_noisy = x+noise
    return x_noisy

syn_band5 = gaussian_noise(syn_band,mu,std5)
syn_band10 = gaussian_noise(syn_band,mu,std10)
syn_band15 = gaussian_noise(syn_band,mu,std15)
syn_band20 = gaussian_noise(syn_band,mu,std20)

# adding the generated data
syn_bandDA = np.vstack((syn_band, syn_band5,syn_band10,syn_band15,syn_band20))

#2. to envelope
env = env.squeeze()
env_std5=0.05*np.std(env)
env_std10=0.1*np.std(env)
env_std15=0.15*np.std(env)
env_std20=0.2*np.std(env)

env_band5 = gaussian_noise(env,mu,env_std5)
env_band10 = gaussian_noise(env,mu,env_std10)
env_band15 = gaussian_noise(env,mu,env_std15)
env_band20 = gaussian_noise(env,mu,env_std20)

# adding the generated data
env_bandDA = np.vstack((env, env_band5,env_band10,env_band15,env_band20))

#3. to phase
#phase = phase[np.newaxis,:,:]
phase_std5=0.05*np.std(phase)
phase_std10=0.1*np.std(phase)
phase_std15=0.15*np.std(phase)
phase_std20=0.2*np.std(phase)

phase_band5 = gaussian_noise(phase,mu,phase_std5)
phase_band10 = gaussian_noise(phase,mu,phase_std10)
phase_band15 = gaussian_noise(phase,mu,phase_std15)
phase_band20 = gaussian_noise(phase,mu,phase_std20)

# adding the generated data
phase_bandDA = np.vstack((phase, phase_band5,phase_band10,phase_band15,phase_band20))

# input file
stack_DA = np.dstack((syn_bandDA, env_bandDA, phase_bandDA))
print(stack_DA.shape)

input_DA = stack_DA.transpose((0,2,1))
#np.save('filelocation', input_DA, allow_pickle=True)  ----------uncomment this line after providing the file path


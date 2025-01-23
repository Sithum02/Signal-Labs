#!/usr/bin/env python
# coding: utf-8

# ## 3.1.1 Implementing Convolution Using Numerical Integration

# In[8]:


from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
h =lambda t: (t > 0)*1.0
x =lambda t: (t > 0) * np.exp(-2*t) # a = −2
Fs = 50 #Sampling frequency for the plotting
T =5 #Timerange
t = np.arange(-T, T, 1/Fs) # Time samples
plt.figure(figsize=(8,3))
plt.plot(t, h(t), label='$h(t)$')
plt.plot(t, x(t), label='$x(t)$')
plt.xlabel(r't$')
plt.legend()
# Plotting
t_ = 1 #For illustration, choose some value for t
flipped = lambda tau: h(t_ -tau)
product = lambda tau: x(tau)*h(t_ -tau)
plt.figure(figsize=(8,3))
plt.plot(t, x(t), label=r'$x(\tau)$')
plt.plot(t, flipped(t), label=r'$h(t - \tau)$')
plt.plot(t, product(t), label=r'$x(\tau)h(t -\tau)$')
# Computing the convolution using integration
y =np.zeros(len(t))
for n, t_ in enumerate(t):
    product = lambda tau: x(tau) * h(t_ - tau)
    y[n] = integrate.simps(product(t), t) # Actual convolution at time t
plt.plot(t, y, label=r'$x(t)\ast h(t)$') # Plotting the output y
plt.xlabel(r'$t$')
plt.legend()


# ## 3.1.2 Convolving with a Signal Composed of Impulse Functions

# In[4]:


fs = 1000 # Sampling frequency for the plotting
delta = lambda t: np.array([fs/10 if 0 < t_ and t_ < 1/(fs/10) else 0.0 for t_ in t])

t = np.arange(-T, T, 1/fs) # Time samples for delta function
# Integration Simpson's rule
y_delta = integrate.simps(delta(t), t)
# Output the calculated value
print(y_delta)


# In[5]:


# Defining new h function
x_2 = lambda t: (t > 0) * np.exp(-2*t) # a = −2
h_2 = lambda t: delta(t+2) + delta(t-1)

fs = 1000 # Sampling frequency for the plotting
T=5 # Timerange
t = np.arange(-T, T, 1/fs) # Time samples

# Computing the convolution using integration by simpson's rule
y = np.zeros(len(t))
for n, t_ in enumerate(t):
    product_2 = lambda tau: x_2(tau) * h_2(t_ - tau) # Get the product
    y[n] = integrate.simps(product_2(t), t) # Actual convolution at time t
    
# Plotting
plt.figure(figsize=(8,3))
plt.plot(t, y, label=r'$x(t)\ast h(t)$') # Plotting the output y
plt.xlabel(r'$t$')
plt.legend();


# ## 3.2 Discrete-Time Systems: Convolution Sum

# In[15]:


# Input signal
x = np.array([0, 1, 1, 2, 0])
# Unit impulse response
h = np.array([0, 0, 0, 3, 1, 0, 0])
hr = np.flip(h)
xo = 2
ho = 4
# Length of the output signal
y = np.zeros(len(x) + len(h) - 1)
for n in range(len(y)):
    xkmin = max(0, n - len(h) + 1)
    xkmax = min(len(x), n + 1)
    hkmin = max(0, len(h) - n -1)
    hkmax = min(len(h), len(x) + len(h) - n - 1)
    y[n] = np.sum(x[xkmin:xkmax]*hr[hkmin:hkmax])
    print("y[{0}] = x[{1}:{2}]*h[{3}:{4}] = {5}".format(n, xkmin, xkmax, hkmin, hkmax, y[n]))


# In[16]:


# Creating the interval [-5, 5]
n = np.arange(-5, 6, 1, dtype=int)

plt.figure(figsize=(8,3))
plt.stem(n,y) # Plotting the output y
plt.xlabel(r'$n$');


# In[17]:


# Input signal
x_2 = np.array([0, 0, 0, 1, 1, 2, 0, 0, 0])
# Impulse response
h_2 = np.array([0, 0, 0, 0, 1, 2, 0, 0, 0])
hr = np.flip(h_2)
xo = 2
ho = 4
# Defining a vector for output signal
y_2 = np.zeros(len(x_2) + len(h_2) - 1)
for n in range(len(y_2)):
    xkmin = max(0, n - len(h_2) + 1)
    xkmax = min(len(x_2), n + 1)
    hkmin = max(0, len(h_2) - n -1)
    hkmax = min(len(h_2), len(x_2) + len(h_2) - n - 1)
    y_2[n] = np.sum(x_2[xkmin:xkmax]*hr[hkmin:hkmax])
    print("y[{0}] = x[{1}:{2}]*h[{3}:{4}] = {5}".format(n, xkmin, xkmax, hkmin, hkmax, y_2[n]))


# In[19]:


# Range of n values [-8, 8]
n = np.arange(-8, 9, 1, dtype=int)
plt.figure(figsize=(8,3))
plt.stem(n, y_2) # Plotting the output y against n
plt.xlabel(r'$n$');
plt.xlim(-4,4); # limiting the n range to [-4, 4]


# In[20]:


from scipy import signal
# Convolving using signal.convolve()
y_full = signal.convolve(x_2, h_2, mode='full')
y_valid = signal.convolve(x_2, h_2, mode='valid')
y_same = signal.convolve(x_2, h_2, mode='same')
# Full
n = np.arange(-8, 9, 1, dtype=int)
plt.figure(figsize=(8,3))
plt.stem(n, y_full) # Plotting the output y for mode='full'
plt.xlabel(r'$n$');
# Valid
plt.figure(figsize=(8,3))
plt.stem(y_valid) # Plotting the output y for mode='valid'
plt.xlabel(r'$n$');
# Same
n = np.arange(-4, 5, 1, dtype=int)
plt.figure(figsize=(8,3))
plt.stem(n, y_same) # Plotting the output y for mode='same'
plt.xlabel(r'$n$');


# ## 3.3 AnApplication in Audio Signal Filtering

# In[4]:


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


data, samplerate = sf.read('anthem.wav')
nyquist = samplerate//2
fc = 2000/nyquist
n =121
b =signal.firwin(n, fc, pass_zero=True)
w, h=signal.freqz(b)


import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w,angles,'g')
ax2.set_ylabel('Angle(radians)',color='g')
ax2.grid()
ax2.axis('tight')


plt.show()


ch1 = signal.convolve(data[:, 0], b, mode='same')
ch2 = signal.convolve(data[:, 1], b, mode='same')


sf.write('audio_file_filtered.wav',np.vstack((ch1,ch2)).T+data,samplerate)


# In[5]:


# Plotting the waveform
fig, ax = plt.subplots(2,1)
ax[0].plot(data);
ax[1].plot(np.vstack((ch1, ch2)).T + data);


# In[ ]:


# Lowpass filter
b, a = signal.cheby1(4, 5, 100, 'low', analog=True)

# Convolution step
ch1 = signal.convolve(data[:, 0], b, mode='same')
ch2 = signal.convolve(data[:, 1], b, mode='same')

# Create output sound file
sf.write('audio_filtered_lowpass.wav', np.vstack((ch1, ch2)).T + data, samplerate)

# Butterworth filter
b, a = signal.butter(4, 100, 'low', analog=True)

# Convolution step
ch1 = signal.convolve(data[:, 0], b, mode='same')
ch2 = signal.convolve(data[:, 1], b, mode='same')

# Create output sound file
sf.write('audio_filtered_butter.wav', np.vstack((ch1, ch2)).T + data, samplerate)


# ## 3.4 Convolution Sum in2-D

# In[10]:


# Input image
x = np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])
# Filter
h = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
# Output image
y = signal.convolve2d(x, h, mode='same')
# Display the output
print(y)
plt.imshow(y);


# In[15]:


# Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
x = mpimg.imread('alan_key.jpg')

x_gray = np.mean(x , axis = 2)

# Display input image
fig, ax = plt.subplots(1,2)
ax[0].imshow(x, cmap='gray') # Show input image

# Convolving filter
h = np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,3]])

# Show output image
ax[1].imshow(signal.convolve2d(x_gray, h, mode='same'), cmap='gray'); # convolution step


# In[ ]:





# In[ ]:





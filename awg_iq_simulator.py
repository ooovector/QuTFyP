from scipy.signal import gaussian
import numpy as np
import tensorflow as tf

class pulses:
	def __init__(self, channels = {}):
		self.channels = channels
		self.settings = {}
	
	## generate waveform of a gaussian pulse with quadrature phase mixin
	def gauss_hd (self, channel, length, amp_x, amp_y, sigma, alpha=0.):
		gauss = gaussian(int(round(length*self.channels[channel].get_clock())), sigma*self.channels[channel].get_clock())
		gauss -= gauss[0]
		gauss_der = np.gradient (gauss)*self.channels[channel].get_clock()
		return amp_x*(gauss + 1j*gauss_der*alpha) + 1j*amp_y*(gauss + 1j*gauss_der*alpha)
		
	## generate waveform of a rectangular pulse
	def rect(self, channel, length, amplitude):
		return amplitude*np.ones(int(round(length*self.channels[channel].get_clock())), dtype=np.complex)

	def pause(self, channel, length):
		return self.rect(channel, length, 0)
		
	def p(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
		
	def ps(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
	
	def set_seq(self, seq, force=True):
		initial_delay = 1e-6
		final_delay = 1e-6
		pulse_seq_padded = [self.p(None, initial_delay, None)]+seq+[self.p(None, final_delay, None)]
	
		pulse_shape = {k:[] for k in self.channels.keys()}
		for channel, channel_device in self.channels.items():
			for pulse in pulse_seq_padded:
				pulse_shape[channel].extend(pulse[channel])
			pulse_shape[channel] = np.asarray(pulse_shape[channel])
	
			if len(pulse_shape[channel])>channel_device.get_nop():
				tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
				tmp = pulse_shape[channel][-channel_device.get_nop():]
				pulse_shape[channel] = tmp
				raise(ValueError('pulse sequence too long'))
			else:
				tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
				tmp[-len(pulse_shape[channel]):]=pulse_shape[channel]
				pulse_shape[channel] = tmp
		
			channel_device.set_waveform(pulse_shape[channel])

class awg_iq_emulator:
    def __init__(self, clock, cutoff, nop, amplitude):
        self.clock = clock
        self.cutoff = cutoff
        self.nop = nop
        self.amplitude = amplitude
        self.resample_valid = False
        self.cdtype = tf.complex64
    def get_clock(self):
        return self.clock
    def get_cutoff(self):
        return self.cutoff
    def get_nop(self):
        return self.nop
    def set_nop(self, cop):
        self.nop = nop
    def set_clock(self, clock):
        self.clock = clock
    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
    
    def set_waveform(self, waveform):
        from scipy.signal import butter, freqs, lfilter
        from scipy.interpolate import interp1d
        self.waveform = waveform
        a,b = butter(1, self.get_cutoff()/self.get_clock(), analog=True)
        #fftfreqs = np.linspace(-self.get_clock()/2, self.get_clock()/2, self.get_nop())
        #w,h=freqs(a,b, fftfreqs)
        self.signal = interp1d(np.arange(self.get_nop())/self.get_clock(), 
                               lfilter(b,a,self.waveform),
                               bounds_error=False,
                               fill_value=None)
        if self.resample_valid:
            self.resample_waveform(self.resample_tstep)
        
    def get_x(self, t, fast=True):
        from scipy.interpolate import interp1d
        if fast and self.resample_valid:
            if type(t) is tf.Tensor:
                index = tf.cast(t/self.resample_tstep, tf.int64)
                return tf.real(self.resampled_waveform[index])
            else:
                index = int(t/self.resample_tstep)
                return np.real(self.resampled_waveform[index])
                
            return tf.real(self.resampled_waveform[index])
            
        if type(t) is tf.Tensor:
            return tf.reshape(tf.py_func(lambda x: np.real(self.signal(x)).astype(np.complex64), [t], self.cdtype), [1])
        else:
            return np.real(self.signal(t))
    def get_y(self, t, fast=True):
        from scipy.interpolate import interp1d
        if fast and self.resample_valid:
            if type(t) is tf.Tensor:
                index = tf.cast(t/self.resample_tstep, tf.int64)
                return tf.real(self.resampled_waveform[index])
            else:
                index = int(t/self.resample_tstep)
                return np.real(self.resampled_waveform[index])
            
        if type(t) is tf.Tensor:
            return tf.reshape(tf.py_func(lambda x: np.imag(self.signal(x)).astype(np.complex64), [t], self.cdtype), [1])
        else:
            return np.imag(self.signal(t))
        
    def resample_waveform(self, tstep):
        from scipy.signal import resample
        self.resampled_valid = True
        self.resampled_tstep = tstep
        self.resampled_waveform = resample(self.waveform, int(self.get_nop()/(tstep*self.get_clock())))
        
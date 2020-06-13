
from scipy.io import wavfile
import numpy as np 
import librosa

class Mfcc():
	def __int__(setf):
		pass
	def pad(seft, signal,w = 2):
		res = []
		for i in range(w):
			res.append(signal[0])
		for i in signal:
			res.append(i)
		for i in range(w):
			res.append(signal[-1])
		return np.array(res)

	def create_paramater(seft,w = 2):
		res = []
		sum_square = 0
		for i in range(-w,0):
			res.append(i*1.0)
			sum_square += i*i
		res.append(0.0)
		for i in range(1,w+1):
			res.append(i*1.0)
			sum_square += i*i
		return sum_square, np.array(res)

	def pad_zero(setf, sig ,sr = 44100):
		thresh = int(0.7 * sr)
		length = sig.shape[0]
		res = sig.copy()
		while (length < thresh):
			res = np.append(res,0)
			length += 1
		return res

	def delta(seft,  signal,w = 2):
		signal = seft.pad(signal,w)
		sum_square, kernel = seft.create_paramater(w = 2)
		i = w
		length = signal.shape[0]
		res = []
		while i + w < length :
			res.append( (signal[i-w: i + w + 1] * kernel).sum()/sum_square )
			i += 1
		return np.array(res)

	def read_audio(setf,path):
		sig,sr = librosa.load(path, sr = 44100)
		return np.array(sig),sr

	def mfcc_10mms(setf,signal, sr ):
		res = []
		mfcc = librosa.feature.mfcc(y=signal, sr=sr,n_mfcc = 12)
		for i in mfcc:
			res.append(i[0])
		deltas_mfcc = setf.delta(signal = res,w = 2) 
		for i in deltas_mfcc:
			res.append(i)
		double_deltas_mfcc = setf.delta(signal = deltas_mfcc, w = 2)
		for i in double_deltas_mfcc:
			res.append(i)
		energy = np.sum( signal.astype(float)**2 )
		res.append(energy)
		deltas_signal = setf.delta(signal= signal , w = 2)
		delta_energy = np.sum( deltas_signal.astype(float)**2 )
		res.append(delta_energy)
		double_deltas_signal = setf.delta(signal= deltas_signal , w = 2)
		doubel_delta_energy = np.sum( double_deltas_signal.astype(float)**2 )
		res.append(doubel_delta_energy)
		return res

	def mfcc(setf,signal,sr):
		signal = setf.pad_zero(signal)
		res = []
		length = signal.shape[0]
		step = sr//100
		i = step
		while i < length:
			res.append( setf.mfcc_10mms(signal[i- step : i] , sr = sr)  )
			i += step
		return np.array(res).T

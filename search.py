
from feature.MFCC import Mfcc
from pre_procesing import *
import numpy as np
import pickle
def search(key):
	res_datas = []
	pd_csv_reader = pd.read_csv('paths_data.csv') 
	for i in range( len(pd_csv_reader)):
		if int (pd_csv_reader['identify'][i]) == key:
			res_datas.append(pd_csv_reader['path'][i])
	return res_datas

model = pickle.load(open('finalized_model.sav', 'rb'))
Mfccer = Mfcc()
path = "Chinh_mot_9.wav"
sig ,sr = Mfccer.read_audio(path)
mfcc = Mfccer.mfcc(sig,sr).T
vector = []
for i in range(0,69):
	for j in range(0,39):
		vector.append(mfcc[i][j])
vector = [vector]
out = model.predict(vector)
searched = search(int (out[0]))

print(f'co tat ca {len(searched)} file')

for i in searched:
	i = i.split('/')[-1]
	print(i)

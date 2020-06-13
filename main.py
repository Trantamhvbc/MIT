from feature.MFCC import Mfcc
from pre_procesing import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def identify_label(path):
	map_ping = {
		"mot":"0",
		"hai":"1",
		"ba":"2",
		"bon":"3",
		"nam":"4",
		"sau":"5",
		"bay":"6",
		"tam":"7",
		"chin":"8",
		"muoi":"9"
	}
	name = path.split('/')[-1]
	
	name = name.split('_')[1]
	return map_ping[name]
def read_txt(path):
	my_file = open(path,'r')
	X = []
	paths = []
	y = []
	while True:
		link_path = my_file.readline()
		if not link_path:
			break
		lenght = int(my_file.readline())
		mfcc = []
		for i in range(lenght):
			line = my_file.readline()
			line_split = line.split()
			for x in line_split:
				mfcc.append(x)
		mfcc = np.array(mfcc).T.astype('float')
		paths.append(link_path)
		X.append(mfcc)
		y.append(identify_label(link_path))
	my_file.close()
	return X,y,paths

def disimilar(x,y):
	if y.shape[-1] < x.shape[-1]:
		return disimilar(y,x)
	lenght_x = x.shape[-1]
	lenght_y = y.shape[-1]
	# print(str(lenght_x) +' ' +str(lenght_y))
	Min = 10000000000
	i = 0 
	while i + lenght_x <= lenght_y:
		tmp = y[:,i:i+lenght_x]
		error = tmp - x
		SE = np.sum(error**2)/(39 * lenght_x)
		if SE < Min:
			Min = SE
		i += 1
	return Min

def extract_audio(path):
	Mfccer = Mfcc()
	sig, sr = Mfccer.read_audio(path)
	return Mfccer.mfcc(signal = sig, sr = sr)




# def main():
# 	datas = read_txt('mfcc.txt')
# 	finder = extract_audio('Chinh_mot_9.wav')
# 	print(finder.shape)
# 	Min = 10000000000
# 	index = "chua co link"
# 	for i in datas:
# 		SE = disimilar(finder,i[1])
# 		if SE < Min:
# 			Min = SE
# 			index = i[0]
# 	print(Min)
# 	print(index)
def main():
	X,y,paths = read_txt('mfcc.txt')
	X_train = X[:700]
	y_train = y[:700]
	X_test = X[700:]
	y_test = y[700:]
	from sklearn import svm
	model = svm.LinearSVC(C = 1)
	model.fit(X,y)
	outs = model.predict(X_test)
	from sklearn.metrics import classification_report
	print(classification_report(y_test,outs))
	import pickle
	pickle.dump(model, open('finalized_model.sav', 'wb'))
	print('save model done')


if __name__ == '__main__':
	main()



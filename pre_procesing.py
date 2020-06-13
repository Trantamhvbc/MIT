import glob
import pandas as pd 
from feature.MFCC import Mfcc

def run():
	path = '/home/dell/Desktop/DATA-MIT/DATA/*.wav'
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
	paths = glob.glob(path)
	res = []
	for path in paths:
		name = path.split('/')[-1]
		print(name)
		name = name.split('_')[1]
		tmp = []
		tmp.append(path)
		tmp.append(map_ping[name])
		res.append(tmp)
	return res
def write_csv(datas,name):
	import csv
	with open(name, mode='w') as csv_file:
		fieldnames = ['path','identify']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		for row in datas:
			row_writer = {}
			row_writer['path'] = row[0]
			row_writer['identify'] = row[1]
			writer.writerow(row_writer)
	print(f'write {name} done') 

def read_csv(path):
	res_datas = []
	pd_csv_reader = pd.read_csv(path) 
	for i in range( len(pd_csv_reader)):
		res_datas.append(pd_csv_reader['path'][i])
	return res_datas
def extract_audio(path):
	Mfccer = Mfcc()
	sig, sr = Mfccer.read_audio(path)
	return Mfccer.mfcc(signal = sig, sr = sr)
def create_data(paths):
	res = []
	c = 0
	for i in paths:
		print(c)
		c += 1 
		res.append( ( i,extract_audio(i) ) )
	return res
def write_mfcc(datas,name = 'mfcc.txt'):
	my_file = open(name,'w')
	for (path, feature) in datas:
		my_file.writelines(path+'\n')
		lenght = feature.shape[-1]
		feature = feature.T
		my_file.writelines( str(lenght) +'\n' )
		for i in feature:
			for j in i:
				my_file.write( str(j) +' ' )
			my_file.write('\n')
	my_file.close()
def static():
	path_csv = 'paths_data.csv'
	paths = read_csv(path_csv)
	Mfccer = Mfcc()
	Max = 0
	for path in paths:
		sig , sr = Mfccer.read_audio(path)
		if sig.shape[0] > Max :
			Max = sig.shape[0]
	return Max
def main():
		path_csv = 'paths_data.csv'
		paths = read_csv(path_csv)
		datas = create_data(paths)
		print(len(datas))
		write_mfcc(datas = datas)
	# write_csv(run(),'paths_data.csv')



if __name__ == '__main__':
	
	main()
	#pass

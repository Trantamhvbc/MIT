import pandas as pd
from scipy.io import wavfile
import librosa
import numpy as np


class readFile:
    def readFileCSV(self,path):
        myFile = pd.read_csv(path)
        listPathAudios = []
        for i in myFile['audio']:
            listPathAudios.append('/home/dell/Desktop/Origin/doan/data/' + i)
            listPathLaybels = []
            for i in myFile['label']:
                listPathLaybels.append('/home/dell/Desktop/Origin/doan/data/' + i)
        return listPathAudios,listPathLaybels

    def readAudio(self,path):
        sr,data = wavfile.read(path)
        return sr,data


    def readAudioByLibrosa(self,path):
        print(path)
        sig = librosa.load(path, sr=16000)
        return sig

    def readAudios(self,listPath):
        Return = []
        for i in listPath:
            tmp = []
            data = readFile.readAudioByLibrosa(self,i)
            tmp.append(data)
            Return.append(tmp)
        return Return

    def readFileLayble(self,path):
        Return = []
        myFile = open(path,'r')
        for line in myFile:
            list = line.split('\t')
            tmp = []
            for j in  range(len(list) - 1):
                tmp.append(float(list[j]))
            Return.append(tmp)
        return Return

    def readFileLaybles(self,listPath):
        Return = []
        for i in listPath:
            Return.append(readFile.readFileLayble(self,i))
        return Return

    def readData(self,path):
        listPathAudios, listPathLabels  = readFile.readFileCSV(self,path)
        dataAudios = []# = readFile.readAudios(self,listPathAudios)
        dataLabels = readFile.readFileLaybles(self,listPathLabels)
        return dataAudios,dataLabels
def detail_audio():
    c = readFile()
    dataAudios, dataLabels = c.readData('/home/dell/Desktop/Origin/doan/data/file.csv')
    max = 0
    min = 100
    k = 0
    sum = 0
    c = 0
    for i in dataLabels:
        k = k + 1
        d = 0
        for j in i:
            d = d + 1
            c = c + 1
            sum = sum + j[1] - j[0]
            if j[1] - j[0] > max:
                max = j[1] - j[0]
            if min > j[1] - j[0]:
                min = j[1] - j[0]
            if (j[1] - j[0] == 2.5817560000000412):
                print('max ', k, " ", d)
            if (j[1] - j[0] == 0.23605400000008103):
                print('min ', k, " ", d)
    print('max ', max)
    print('min ', min)
    print('trung binh ', sum / c)
    print('sum ', c)
    max = 0
    min = 10000
    k = 0
    sum = 0
    c = 0
    for i in dataLabels:
        k = k + 1
        d = 0
        for j in range(len(i) - 1):
            d = d + 1
            c = c + 1
            sum = sum + i[j + 1][0] - i[j][1]
            if i[j + 1][0] - i[j][1] > max:
                max = i[j + 1][0] - i[j][1]
            if min > i[j + 1][0] - i[j][1]:
                min = i[j + 1][0] - i[j][1]
            if (i[j + 1][0] - i[j][1] == 2.5817560000000412):
                print('max ', k, " ", d)
            if (i[j + 1][0] - i[j][1] == 0.0):
                print('min ', k, " ", d)
    print('max ', max)
    print('min ', min)
    print('trung binh ', sum / c)
    print('sum ', c)

# detail_audio()
class DeterminedLabel:
    def low_binary_search_by_start(self, data, key):
        l = 0
        r = len(data) - 1
        res = -1
        while l <= r:
            mid = int( (l+r)/2 )
            if  data[mid][0] <= key:
                res = mid
                l = mid + 1
            else:
                r = mid - 1
        return res

    def up_binary_search_by_end(self, data, key):
        l = 0
        r = len(data) - 1
        res = r + 1
        while l <= r:
            mid = int( (l+r)/2 )
            if  key <= data[mid][1]:
                res = mid
                r = mid - 1
            else:
                l = mid +1
        return res

    def determined_laybel_by_begin_end_time(self,begin,end , label):
        i_start = self.low_binary_search_by_start(data= label,key=begin)
        i_end = self.up_binary_search_by_end(data= label, key=end)
        time_snore = 0
        if begin <= label[i_start][1]:
            time_snore = time_snore + label[i_start][1] - begin
        i_start = i_start + 1
        if end >= label[i_end][0]:
            time_snore = time_snore + end - label[i_end][0]
        i_end = i_end - 1
        while i_start <= i_end:
            time_snore = time_snore + label[i_start][1] - label[i_start][0]
            i_start = i_start + 1
        if 2*time_snore + begin >= end:
            return  1
        else:
            return 0


    def determinedLabel(self,data,label,sr):
        end = 3*sr
        start = 0
        length = data.shape
        data_return = []
        label_return = []
        while end < length:
            data_return.append(data[start:end])
            label_return.append(self.determined_laybel_by_begin_end_time(begin=start,end=end,label=label))
        return np.array(data), np.array(label_return)



data = [ [1,2],
         [3,4],
         [5,6],
         [7,8],
         [9,10],
         [11,12],
         [14,15],
         [16,17],
         [18,19]]

d = DeterminedLabel()
print(d.up_binary_search_by_end(data=data,key=6.6))
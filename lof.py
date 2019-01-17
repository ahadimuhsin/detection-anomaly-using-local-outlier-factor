import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import csv

'''
membaca file data.txt
di dalam file tersebut, terdapat angka yang digunakan untuk mencari LOF
'''
f = open ("F:/Kuliah/Semester 7/Data Mining/Praktikum/8/soal_uas.csv", "r")
#mengubah data menjadi list
data_jarak = list (csv.reader(f))
print (data_jarak)
#mengubah list menjadi array dan bertipe float
data = np.array(data_jarak).astype(np.float) 
# print (data) ---> untuk pengecekan nilai


print("LOCAL OUTLIER FACTOR")
print("----------------------------------------------------------------------")
print ("Menghitung Euclidean Distance")
#mencari euclidean distance dengan library squareform
euclidean_distance = squareform(pdist(data, 'euclidean'))
print(euclidean_distance)
print(" ")

#Mengurutkan jarak hasil euclidean distance 
print ("Mengurutkan jarak terdekat")
euclid = np.array(euclidean_distance) 
ind = np.argsort(euclid, axis=1)
# print ("lalalala", ind)
euclid_sorted = data[ind]
hasil_sort_euclid = np.sort(euclid, axis=1)
print(hasil_sort_euclid)
print(" ")

#masukkan berapa banyak tetangga terdekat berdasarkan hasil perhitungan euclidean
print ("Masukkan berapa banyak tetangga terdekat")
neighbor = int (input('Banyak tetangga: '))
print ("Menentukan", neighbor ," tetangga terdekat") 
d = hasil_sort_euclid
e = np.array(d)

k = []
for key, value in enumerate(e):
    n = []
    for j, w in enumerate(value):
        if( j >= 1 and j <= neighbor ) :
            n.append(w)
    k.append(n)
k = np.array(k)
print(k)
print(" ")

#menghitung density data
print("Menghitung density")
dens = []
for key, value in enumerate(k) :
    dens.append((np.sum(value)/neighbor)**-1)
dens = np.array(dens)
print(dens) 
print(" ")

#mengurutkan density sesuai urutan data
print("Hasil density sesuai urutan data")
dens_r = [] 
for key, value in enumerate(k) :
    data_row = euclidean_distance[key].tolist()
    dens_f = []
    for kunci, nilai in enumerate(value) :
        index = data_row.index(nilai)
        dens_f.append(dens[index])
    dens_r.append(dens_f)
dens_r = np.array(dens_r)
print(dens_r)
print(" ")

print("Menghitung LOF dari tiap data")
lof = []
for key, value in enumerate(dens_r) :
    l = np.sum(value)
    lof.append((l/neighbor)/dens[key])
lof = np.array(lof)
print(lof)
print (" ")

# print("Grafik Persebaran Data")
threshold = 2
for x in range(len(lof)):
    if lof[x]<threshold:
         plt.plot(data[x][0],data[x][1],"bo")
    else:
        plt.plot(data[x][0],data[x][1],"ro")
plt.title ("Grafik Local Outlier Factor")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Outlier', 'Inlier'], loc = 'upper left', frameon = 'true')
plt.show()

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as spc
import math
from scipy.special import lpmn
import scipy.integrate as integrate
from scipy.integrate import quad
from numpy import sin, cos

# Set the number of sources and the coordinates for the input


nsources = int(1297803)
nside = 8
npix = hp.nside2npix(nside)

# Coordinates and the density field f
#thetas = np.random.random(nsources) * np.pi
#phis = np.random.random(nsources) * np.pi * 2.

#fs = np.random.randn(nsources)

with open("eventFileTotal.txt") as inputFile2:
   lines2 = inputFile2.readlines()

thetas2 = []
phis2 = []

for i in range(nsources):
   thetas2.append(float(lines2[i+1].split()[1]))
   phis2.append(float(lines2[i+1].split()[2]))

indices2 = hp.ang2pix(nside, thetas2, phis2)

hpxmap2 = np.zeros(npix, dtype = np.float)
for i in range(nsources):
    hpxmap2[indices2[i]]+= 1.0
    #hpxmap2[indices2[i]] += npix*(1.0/nsources)




with open("eventFileSingle.txt") as inputFile:
    firstLine = inputFile.readline()
    lines = inputFile.readlines()
    #print (lines[1].split()[1])

cl = []
for l in range(24):
   cl.append(0)

events = int(firstLine)
#print(len(lines))
i = 0
for x in range(events):
 
    i += 1
    j = x+1
    phis = []
    thetas = []
    while i < len(lines) and len(lines[i].split()) >2  and float(lines[i].split()[0]) != j:
      #print(lines[i+1].split()[0])
      thetas.append(float(lines[i].split()[1]))
      phis.append(float(lines[i].split()[2]))
      i+=1

    indices = hp.ang2pix(nside, thetas, phis)
    hpxmap = np.zeros(npix, dtype = np.float)
    for k in range(len(thetas)):
      hpxmap[indices[k]] += 1.0
      #hpxmap[indices[k]] += npix*(1.0/len(thetas))
    hpxmapFinal = np.zeros(npix, dtype = np.float)
    for v in range(len(thetas2)):
      hpxmapFinal[indices2[v]] += (hpxmap[indices2[v]]/hpxmap2[indices2[v]]) * (npix*(1.0/len(thetas)))
    c = hp.anafast(hpxmapFinal)
    print(c)
    print("") 
    print("")

DPI = 100
SIZE = 400 

hp_smoothed = hp.sphtfunc.smoothing(hpxmap, fwhm=np.radians(3.), iter = 1)

hp.mollview(hp_smoothed, cmap = cm.jet, norm = "hist", xsize = SIZE, title='Real Data smoothed')
plt.savefig("AMPT_smoothed1.png", dpi = DPI)
hp.graticule()

print(c)
print(hp.anafast(hpxmap))
print(hp.anafast(hpxmap2))
'''
    alm = hp.map2alm(hpxmapFinal, lmax = 23)
    cl2 = []
    
    for g in range(len(c)):
       cl2.append(c[g] - (1.0/(2*g+1)*((abs(alm[g]))**2)))

    for z in range(len(c)):
      cl[z] = cl[z] + cl2[z]
for c in range(len(cl)):
    cl[c] = cl[c] / (1.0*events)

#print("")
#print("")
#print("")
#print("")
#print("")
#print("")
#print("")


nsources = int(199999)
nside = 8
npix = hp.nside2npix(nside)

with open("eventFileT_iso.txt") as inputFile4:
   lines4 = inputFile4.readlines()

thetas4 = []
phis4 = []

for i in range(nsources):
   thetas4.append(float(lines4[i+1].split()[1]))
   phis4.append(float(lines4[i+1].split()[2]))

indices4 = hp.ang2pix(nside, thetas4, phis4)

hpxmap4 = np.zeros(npix, dtype = np.float)
for i in range(nsources):
    hpxmap4[indices4[i]] += npix*(1.0/nsources)

 


with open("eventFile_iso.txt") as inputFile3:
    firstLine2 = inputFile3.readline()
    lines3 = inputFile3.readlines()
    #print (lines[1].split()[1])
c2 = []
for l in range(24):
   c2.append(0)
events2 = int(firstLine2)
#print(len(lines))
i = 0
for x in range(events2):
    i += 1
    j = x+1
    phis3 = []
    thetas3 = []
    while i < len(lines3) and len(lines3[i].split()) >2  and float(lines3[i].split()[0]) != j:
      #print(lines[i+1].split()[0])
      thetas3.append(float(lines3[i].split()[1]))
      phis3.append(float(lines3[i].split()[2]))
      i+=1

    indices3 = hp.ang2pix(nside, thetas3, phis3) 
    hpxmap3 = np.zeros(npix, dtype = np.float)   
    for k in range(len(thetas3)):
      hpxmap3[indices3[k]] += npix*(1.0/len(thetas3))
    hpxmapFinal2 = np.zeros(npix, dtype = np.float)
    for v in range(len(thetas3)):
      hpxmapFinal2[indices3[v]] += hpxmap3[indices3[v]]/hpxmap4[indices3[v]]
    c2f = hp.anafast(hpxmapFinal2)



    alm2 = hp.map2alm(hpxmapFinal2, lmax = 23)
    cl2 = []

      

    for g in range(len(c2f)):
       cl2.append(c2f[g] - (1.0/(2*g+1)*((abs(alm2[g]))**2)))
    #print(alm2)
    for z in range(len(cl2)):
      c2[z] = c2[z] + cl2[z]   
for c in range(len(cl)):
    c2[c] = c2[c] / (1.0*events2)


for i in range(len(cl)):
   cl[i] = cl[i] - c2[i]
'''
'''
plt.yscale('log')
plt.plot(c)
plt.savefig("powerspect_AMPT.png")
'''
'''
b11 = math.sqrt((2*1+1)/(4*math.pi)*1/math.factorial(2))* -1.56727

b00 = math.sqrt((2*0+1)/(4*math.pi)*1/math.factorial(0))* 1.96962

b22 = math.sqrt((2*2+1)/(4*math.pi)*1/math.factorial(4))* 3.99862

b33 = math.sqrt((2*3+1)/(4*math.pi)*1/math.factorial(6))* -17.6705




v1 = (3/2)* cl[1]/(abs(b11)**2)*(abs(b00))**2/(4*math.pi)
v1F = math.sqrt(abs(v1))
print(v1F)

v2 = (5/2)*cl[2]/(abs(b22)**2)*(abs(b00))**2/(4*math.pi)
v2F = math.sqrt(abs(v2))
print(v2F)

v3 = (7/2)*cl[3]/(abs(b33)**2)*(abs(b00))**2/(4*math.pi)
v3F = math.sqrt(abs(v3))

print(v3F)

'''
'''
cOdd = []

for c in range(11):
   cOdd.append(cl[c*2+1])
plt.xscale('log')
plt.yscale('log')
plt.plot(cOdd)
plt.savefig("powerspect_ODD.png")
    

# Go from HEALPix coordinates to indices
indices = hp.ang2pix(nside, thetas, phis)

# Initate the map and fill it with the values
hpxmap = np.zeros(npix, dtype=np.float)
for i in range(nsources):
    #hpxmap[indices[i]] += fs[i]
    hpxmap[indices[i]] += npix*(1.0/nsources)

DPI = 100
SIZE = 400

# Inspect the map
#plt.figure(1)
'''
#map_ring = hp.pixelfunc.reorder(hpxmap, inp = 'NEST', out = 'RING')
#hp.mollview(hpxmap, xsize = SIZE)
'''
hp_smoothed = hp.sphtfunc.smoothing(hpxmap, fwhm=np.radians(5.), iter = 1)

hp.mollview(hp_smoothed, cmap = cm.jet, norm = "hist", xsize = SIZE, title='Real Data smoothed')
plt.savefig("real_data_smoothed.png", dpi = DPI)
hp.graticule()
'''
'''
cl = hp.anafast(hpxmap,lmax=23)
plt.yscale('log')
axes = plt.gca()
axes.set_ylim([0.001,100])
plt.plot(cl)
plt.savefig('powerspect.png')
print(cl)
'''


'''
#plt.figure(2)
# Get the power spectrum
Cl = hp.anafast(hpxmap)
#print(Cl)
plt.plot(Cl)
plt.ylabel('C_{l}')
plt.savefig('plot_toyModel_power_spectrum.png')
'''


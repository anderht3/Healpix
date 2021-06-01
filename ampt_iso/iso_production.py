import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.special as spc
import math
import matplotlib as mpl
from scipy.special import lpmn
import scipy.integrate as integrate
from scipy.integrate import quad
from numpy import sin, cos
from matplotlib.cm import ScalarMappable
import random
# Set the number of sources and the coordinates for the input


nsources = int(199999)
nside = 8
npix = hp.nside2npix(nside)
SIZE = 400
DPI = 100
# Coordinates and the density field f
#thetas = np.random.random(nsources) * np.pi
#phis = np.random.random(nsources) * np.pi * 2.

#fs = np.random.randn(nsources)

'''
with open("eventFileFullTot_iso.txt") as inputFile2:
   lines2 = inputFile2.readlines()
thetas2 = []
phis2 = []
for i in range(nsources):
   thetas2.append(float(lines2[i+1].split()[1]))
   phis2.append(float(lines2[i+1].split()[2]))
indices2 = hp.ang2pix(nside, thetas2, phis2)
'''
clFinal = np.zeros(24)
cl2Final = np.zeros(24)
clSqr = np.zeros(24)
clErr = np.zeros(24)
cl2Sqr = np.zeros(24)
cl2Err = np.zeros(24)


indices2 = []
hpxmap2 = np.zeros(npix, dtype = np.float)
events = 10002
mult = 7674 

print(hp.ang2pix(nside,0.174533,np.pi))
print(hp.ang2pix(nside,0.174533,2*np.pi))
print(hp.ang2pix(nside,2.96706,np.pi))
print(hp.ang2pix(nside,2.96706,2*np.pi))
print(npix)
#arr = [[0]*(mult)]*events
#arr = [ [0]*mult for i in range(events)]
arr = [[0 for i in range(mult)] for j in range(events)] 
for i in range(events):
    #if i%50 == 0:
        #print("First loop with i = "+str(i)+', and events = '+str(events))
    #print(i)
    #hpxmap2 = np.zeros(npix, dtype = np.float)
    for k in range(mult):
       ipix = random.randint(4,760)
       #if k == 0:
       #print('i = '+str(i)+'   ipix = '+str(ipix))
       #hpxmap2[indices2[i]] += 1.0
       hpxmap2[ipix] += 1
       #indices2.append(ipix)
       arr[i][k] = hp.pix2ang(nside,ipix)
       #hpxmap[ipix] += 1
       #print('i = '+str(i)+', First loop, angles: '+str(hp.pix2ang(nside,ipix)))
       #print('i = '+str(i)+', First loop, angles: '+str(arr[i][k]))


hpxmapFinal = np.zeros(npix, dtype = np.float)
for i in range(events): 
    #if i%50 == 0:
        #print('Second loop with i = '+str(i)+', and events = '+str(events))
    c = np.zeros(24)   
    hpxmapFinal = np.zeros(npix, dtype = np.float)
    hpxmap = np.zeros(npix, dtype = np.float)
    for k in range(mult):
       #print('arr[i][k] = '+str(arr[i][k]))
       indices = hp.ang2pix(nside,arr[i][k][0],arr[i][k][1])
       #print('i = '+str(i)+', indices = '+str(indices))
       #indices2.append(indices)
       hpxmap[indices] += 1

    for n in range(npix):
       if (hpxmap2[n] > 0):
          hpxmapFinal[n] += (hpxmap[n]/hpxmap2[n]) * (events)
       #if l == 0:
       #    print('i = '+str(i)+'   theta = '+str(arr[i][l][0]))

    c = hp.anafast(hpxmapFinal)
    alm = hp.map2alm(hpxmapFinal, lmax = 23)

    cl2 = []
    for g in range(24):
     cl2.append(c[g] - (1.0/(2*g+1)*((abs(alm[g]))**2)))

    #print(c)
    for l in range(len(cl2)):
        clFinal[l] = clFinal[l] + c[l]
        cl2Final[l] = cl2Final[l] + cl2[l]
        clSqr[l] = clSqr[l] + c[l]*c[l]
        cl2Sqr[l] = cl2Sqr[l] + cl2[l]*cl2[l]

    #print(c)

for k in range(len(cl2)):
   clFinal[k] = clFinal[k] / (1.0*events)
   cl2Final[k] = cl2Final[k] / (1.0*events)
   cl2Sqr[k] = cl2Sqr[k]/ (1.0*events)
   cl2Err[k] = math.sqrt((cl2Sqr[k] - cl2Final[k]*cl2Final[k])/(1.0*events))
   clSqr[k] = clSqr[k]/ (1.0*events)
   clErr[k] = math.sqrt((clSqr[k] - clFinal[k]*clFinal[k])/(1.0*events))
   print(k,clFinal[k],clErr[k])
#print()
#print()
#print(cl2Final)
print()

for k in range (24):
   print(k,cl2Final[k],cl2Err[k])


plt.yscale('log')
plt.plot(clFinal)
plt.savefig("powerspect_iso2.png")
#hpxmap2[1] = 50
#hpxmap2[npix-2] = 150

#hp_smoothed = hp.sphtfunc.smoothing(hpxmap2, fwhm=np.radians(1), iter = 1)
'''
hp.mollview(hpxmap2, cmap = cm.jet, xsize = SIZE, min = events*mult/npix*0.9, max = events*mult/npix*1.1, title='Isotropic randomised')
#fig = plt.gcf()
#ax = plt.gca()
#norm = mpl.colors.Normalize(vmin = 50, vmax = 150)
#image = ax.get_images()[0]
#cmap = fig.colorbar(image, norm = norm, orientation = 'horizontal')
hp.graticule()
plt.savefig("map_iso.png", dpi = DPI)
'''
'''
cl = []
for l in range(24):
   cl.append(0)
i = 0
events = 100
for x in range(events):
 
    i += 1
    j = x+1
    phis = []
    thetas = []
      #print(lines[i+1].split()[0])
      thetas.append(float(lines[i].split()[1]))
      phis.append(float(lines[i].split()[2]))
      i+=1
    indices = hp.ang2pix(nside, thetas, phis)
    hpxmap = np.zeros(npix, dtype = np.float)
    
    for k in range(len(thetas)):
      hpxmap[indices[k]] += 1.0
'''

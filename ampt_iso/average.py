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


with open("ampt0_1m.txt") as inputFile1:
   lines = inputFile1.readlines()

with open("ampt0_2m.txt") as inputFile2:
   lines2 = inputFile2.readlines()

with open("ampt0_3m.txt") as inputFile3:
   lines3 = inputFile3.readlines()

with open("ampt0_4m.txt") as inputFile4:
   lines4 = inputFile4.readlines()

with open("ampt0_5m.txt") as inputFile5:
   lines5 = inputFile5.readlines()
total = np.zeros(24)
totalErr = np.zeros(24)

for i in range(24):
   total[i] += (float(lines[i].split()[1])+float(lines2[i].split()[1])+float(lines3[i].split()[1])+float(lines4[i].split()[1])+float(lines5[i].split()[1]))/5.0
   totalErr[i] += (float(lines[i].split()[2])+float(lines2[i].split()[2])+float(lines3[i].split()[2])+float(lines4[i].split()[2])+float(lines5[i].split()[2]))/5.0
   print(i,total[i],(totalErr[i]/math.sqrt(5)))

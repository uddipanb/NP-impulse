import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import random
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.pylab import *
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.ticker as ticker
from scipy import integrate
from scipy import optimize
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from scipy.special import erf as errorfunc
from scipy.special import gamma, factorial
from matplotlib import cm
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.cosmology import z_at_value
from numpy import linalg
import time
import os
import h5py
from matplotlib.animation import FuncAnimation

#-----------------------------------------------------------------

Np=1000000 #100000,1000000
Nrand=10000

#-----------------------------------------------------------------

def Psig(rg):
    f1=1/np.sqrt(1+rg**2) #Plummer sphere
    #f1=1/(1+rg)            #Hernquist sphere
    #f1=1/rg		    #Pt mass
    #f1=phi_spline(rg)              #Simulation
    return f1

def rhog(rg):
    f1=(3/(4*np.pi))*(1/(1+rg**2)**2.5)    
    return f1

def mass_enc(rg):
    return rg**3/(1+rg**2)**1.5 #Plummer

def fdist(e):
    f=(3/(7*np.pi**1.5))*(2*abs(e))**3.5
    return f

#----------------------------------------------------------------------

#Initial conditions

x=np.zeros(Np)                 
y=np.zeros(Np)
z=np.zeros(Np)
vx=np.zeros(Np)
vy=np.zeros(Np)
vz=np.zeros(Np)
Eg=np.zeros(Np)

Rmax=20

Mmin=0
Mmax=1
#Mmax=(Rmax**3)/(1+Rmax**2)**1.5

costhetamin=-1
costhetamax=1
phimin=0
phimax=2*np.pi

vmin=0
costhetavmin=-1
costhetavmax=1
phivmin=0
phivmax=2*np.pi

start=time.time()

for i in range(Np):


    for j in range(Nrand):

        randr=np.random.choice(a=Nrand+1, size=1)    
        Mratio=Mmin+(randr[0]/Nrand)*(Mmax-Mmin)
        
        Menc=Mratio*((1+Rmax**2)**1.5/Rmax**3)
        rg=np.sqrt(1/(Menc**(-2.0/3)-1))

        if (rg<=Rmax):
            break


    randcostheta=np.random.choice(a=Nrand+1, size=1)   
    costheta=costhetamin+(randcostheta[0]/Nrand)*(costhetamax-costhetamin)
    sintheta=np.sqrt(1-costheta**2)

    randphi=np.random.choice(a=Nrand+1, size=1)
    phi=phimin+(randphi[0]/Nrand)*(phimax-phimin)
    
    zr=rg*costheta
    xr=rg*sintheta*np.cos(phi)
    yr=rg*sintheta*np.sin(phi)

    
    vesc=np.sqrt(2*Psig(rg))

    
    for j in range(Nrand):

        randv=np.random.choice(a=Nrand+1, size=1)
        vr=vmin+(randv[0]/Nrand)*(vesc-vmin)
        e=Psig(rg)-0.5*vr**2

        randf=np.random.choice(a=Nrand+1, size=1)

        fmin=0
        v_fmax=(2/3.0)*np.sqrt(Psig(rg))
        e_fmax=Psig(rg)-0.5*v_fmax**2
        fmax=(v_fmax**2)*fdist(e_fmax)

        fr=fmin+(randf[0]/Nrand)*(fmax-fmin)

        if (fr<=vr**2*fdist(e)):            
            break

    randcosthetav=np.random.choice(a=Nrand+1, size=1)   
    costhetav=costhetavmin+(randcosthetav[0]/Nrand)*(costhetavmax-costhetavmin)
    sinthetav=np.sqrt(1-costhetav**2)

    randphiv=np.random.choice(a=Nrand+1, size=1)
    phiv=phivmin+(randphiv[0]/Nrand)*(phivmax-phivmin)        

    vzr=vr*costhetav
    vxr=vr*sinthetav*np.cos(phiv)
    vyr=vr*sinthetav*np.sin(phiv)

    x[i]=xr
    y[i]=yr
    z[i]=zr
    vx[i]=vxr
    vy[i]=vyr
    vz[i]=vzr
    Eg[i]=abs(e)

    print (i)
    end=time.time()
    dt=end-start
    print ("Time taken: %f s" %(dt))
    
#---------------------------------------------------------------------

hf=h5py.File('particle_data_plummer.h5','w')
hf.create_dataset('x',data=x)
hf.create_dataset('y',data=y)
hf.create_dataset('z',data=z)
hf.create_dataset('vx',data=vx)
hf.create_dataset('vy',data=vy)
hf.create_dataset('vz',data=vz)
hf.close()

#---------------------------------------------------------------------


#Testing

dr=0.0001  #0.07
r0=0.5

nbins=120
num_eg=np.zeros(nbins)
eg=np.linspace(0,Psig(r0),nbins+1)
del_eg=(eg.max()-eg.min())/nbins

rg=np.sqrt(x**2+y**2+z**2)
for i in range(nbins):
    mask1=rg>r0-dr/2
    x1=x[mask1]
    y1=y[mask1]
    z1=z[mask1]
    Eg1=Eg[mask1]
    rg1=np.sqrt(x1**2+y1**2+z1**2)
    mask2=rg1<r0+dr/2
    x2=x1[mask2]
    y2=y1[mask2]
    z2=z1[mask2]
    Eg2=Eg1[mask2]
    mask3=Eg2>eg[i]
    Eg3=Eg2[mask3]
    mask4=Eg3<eg[i+1]
    Eg4=Eg3[mask4]
    num_eg[i]=len(Eg4)

eg1=np.linspace(0,Psig(r0),100)
f=np.zeros(len(eg1))
for i in range(len(eg1)):
    f[i]=(16*np.pi**2*r0**2)*np.sqrt(2*(Psig(r0)-eg1[i]))*fdist(eg1[i])*(dr*del_eg)*Np
eg2=np.linspace(0,Psig(r0),nbins)

fig1,ax1=plt.subplots()
ax1.plot(eg2,num_eg,color='b')
ax1.plot(eg1,f,color='k',ls='--')




Nr=20
r=np.logspace(-1,np.log10(Rmax),num=Nr)
#r=np.linspace(0.1,Rmax,Nr)
dr=r[1]-r[0]
#dr=0.001
density_recovered=np.zeros(Nr)
density=np.zeros(Nr)

start=time.time()

for i in range(Nr):

    counter=0
    for j in range(Np):
        rg=np.sqrt(x[j]**2+y[j]**2+z[j]**2)
        #if (r[i]-dr/2<rg<r[i]+dr/2):
        if (rg<=r[i]):
            counter+=1

    #density_recovered[i]=(counter/(4*np.pi*r[i]**2*dr))/Np
    density_recovered[i]=counter/Np
    #density[i]=rhog(r[i])
    density[i]=mass_enc(r[i])

    print (i)
    end=time.time()
    dt=end-start
    print ("Time taken: %f s" %(dt))

fig2,ax2=plt.subplots()
ax2.semilogx(r,density_recovered,color='b')
ax2.semilogx(r,density,color='r')

plt.show()




























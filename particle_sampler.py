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

Np=100000        #No of subject stars (1000000)
Nrand=100000      #Maximum no of random realizations for each star
EPS=1e-8          #1e-6

#-----------------------------------------------------------------

#--------Functions--------

def Psig(rg):
    #f1=1/np.sqrt(1+rg**2)                #Plummer sphere
    f1=1/(1+rg)                           #Hernquist sphere
    return f1

def rhog(rg):
    #f1=(3/(4*np.pi))*(1/(1+rg**2)**2.5)  #Plummer sphere
    f1=(1/(2*np.pi))*(1/(rg*(1+rg)**3))   #Hernquist sphere
    return f1

def mass_enc(rg):
    #f1=rg**3/(1+rg**2)**1.5              #Plummer sphere
    f1=rg**2/(1+rg)**2                    #Hernquist sphere
    return f1

def fdist(e):
    eps=abs(e)
    #f=(3/(7*np.pi**1.5))*(2*eps)**3.5    #Plummer sphere
    f=(1/(8*np.sqrt(2)*np.pi**3))*((3*np.arcsin(np.sqrt(eps))+np.sqrt(eps*(1-eps))*(1-2*eps)*(8*eps**2-8*eps-3))/(1-eps)**2.5)                                #Hernquist sphere
    return f

Ne=100000
energy=np.logspace(np.log10(EPS),np.log10(1-EPS),num=Ne)
loge=np.log10(energy)

FDIST=fdist(energy)
logFDIST=np.log10(FDIST)

Fspline=CubicSpline(loge,logFDIST)
Fspline_deriv=Fspline.derivative(nu=1)
FDIST_DERIV=(FDIST/energy)*Fspline_deriv(loge)

def max_energy_finder(E,PSI):
    logE=np.log10(E)
    f1=fdist(E)
    df1dE=(f1/E)*Fspline_deriv(logE)
    return f1/df1dE+E-PSI

def max_energy(PSI):
    return optimize.brentq(max_energy_finder,EPS,1-EPS,args=(PSI))

#----------------------------------------------------------------------

#--------Sampling positions and velocities of subject stars--------

x=np.zeros(Np)                 
y=np.zeros(Np)
z=np.zeros(Np)
vx=np.zeros(Np)
vy=np.zeros(Np)
vz=np.zeros(Np)
Eg=np.zeros(Np)

Rmax=5000              #Truncation radius of subject

Mmin=0
Mmax=1

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


    for j in range(Nrand):                                                     #Sampling r

        #randr=np.random.choice(a=Nrand+1, size=1)    
        #Mratio=Mmin+(randr[0]/Nrand)*(Mmax-Mmin)
        randr=np.random.uniform(0,1)
        Mratio=Mmin+randr*(Mmax-Mmin)
        
        #Menc=Mratio*((1+Rmax**2)**1.5/Rmax**3)                                #Plummer
        #rg=np.sqrt(1/(Menc**(-2.0/3)-1))

        #Menc=Mratio*((1+Rmax)**2/Rmax**2)                                      #Hernquist
        Menc=Mratio
        
        rg=1/(Menc**(-0.5)-1)

        if (rg<=Rmax):
            break


    #randcostheta=np.random.choice(a=Nrand+1, size=1)                           #Sampling position angles
    #costheta=costhetamin+(randcostheta[0]/Nrand)*(costhetamax-costhetamin)
    randcostheta=np.random.uniform(0,1)
    costheta=costhetamin+randcostheta*(costhetamax-costhetamin)
    sintheta=np.sqrt(1-costheta**2)

    #randphi=np.random.choice(a=Nrand+1, size=1)
    #phi=phimin+(randphi[0]/Nrand)*(phimax-phimin)
    randphi=np.random.uniform(0,1)
    phi=phimin+randphi*(phimax-phimin)
    
    zr=rg*costheta
    xr=rg*sintheta*np.cos(phi)
    yr=rg*sintheta*np.sin(phi)

    
    vesc=np.sqrt(2*Psig(rg))                                                   #Escape velocity at r

    
    for j in range(Nrand):                                                     #Sampling v

        #randv=np.random.choice(a=Nrand+1, size=1)
        #vr=vmin+(randv[0]/Nrand)*(vesc-vmin)
        randv=np.random.uniform(0,1)
        vr=vmin+randv*(vesc-vmin)
        e=Psig(rg)-0.5*vr**2

        #randf=np.random.choice(a=Nrand+1, size=1)
        randf=np.random.uniform(0,1)

        fmin=0
        #v_fmax=(2/3.0)*np.sqrt(Psig(rg))                                      #Plummer
        #e_fmax=Psig(rg)-0.5*v_fmax**2
        
        
        e_fmax=max_energy(Psig(rg))                                            #Hernquist
        v_fmax=np.sqrt(2*(Psig(rg)-e_fmax))        

        fmax=(v_fmax**2)*fdist(e_fmax)

        #fr=fmin+(randf[0]/Nrand)*(fmax-fmin)
        fr=fmin+randf*(fmax-fmin)

        if (fr<=vr**2*fdist(e)):            
            break

    #randcosthetav=np.random.choice(a=Nrand+1, size=1)                          #Sampling velocity angles
    #costhetav=costhetavmin+(randcosthetav[0]/Nrand)*(costhetavmax-costhetavmin)
    randcosthetav=np.random.uniform(0,1)
    costhetav=costhetavmin+randcosthetav*(costhetavmax-costhetavmin)
    sinthetav=np.sqrt(1-costhetav**2)

    #randphiv=np.random.choice(a=Nrand+1, size=1)
    #phiv=phivmin+(randphiv[0]/Nrand)*(phivmax-phivmin)        
    randphiv=np.random.uniform(0,1)
    phiv=phivmin+randphiv*(phivmax-phivmin)

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

#--------Writing data in a file--------

#hf=h5py.File('particle_data_plummer.h5','w')              #Plummer
#hf=h5py.File('particle_data_hernquist.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_rmax1e1.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_rmax1e4.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_rmax1e5.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_rmax1e6.h5','w')             #Hernquist

#hf=h5py.File('../data/particle_data_hernquist_Np1e5_rmax1e5_r20.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_Np1e5_rmax1e4_r3.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_Np1e5_rmax5e4_r1.h5','w')             #Hernquist
#hf=h5py.File('../data/particle_data_hernquist_Np1e5_rmax1e3_r20.h5','w')             #Hernquist
hf=h5py.File('../data/particle_data_hernquist_Np1e5_rmax5e3_r20.h5','w')             #Hernquist
hf.create_dataset('x',data=x)
hf.create_dataset('y',data=y)
hf.create_dataset('z',data=z)
hf.create_dataset('vx',data=vx)
hf.create_dataset('vy',data=vy)
hf.create_dataset('vz',data=vz)
hf.close()


#---------------------------------------------------------------------

#--------Testing--------

dr=0.07  #0.07
r0=2

nbins=20
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
#ax1.hist(num_eg,bins=eg2,color='b')
ax1.plot(eg1,f,color='k',ls='--')




Nr=10 #20
r=np.logspace(-1,np.log10(Rmax),num=Nr)
#r=np.linspace(0.1,Rmax,Nr)
dr=r[1]-r[0]
#dr=0.001
density_recovered=np.zeros(Nr)
density=np.zeros(Nr)
massenc_recovered=np.zeros(Nr)
massenc=np.zeros(Nr)

start=time.time()

for i in range(Nr):

    counter=0
    for j in range(Np):
        rg=np.sqrt(x[j]**2+y[j]**2+z[j]**2)
        #if (r[i]-dr/2<rg<r[i]+dr/2):
        if (rg<=r[i]):
            counter+=1

    #density_recovered[i]=(counter/(4*np.pi*r[i]**2*dr))/Np
    massenc_recovered[i]=counter/Np
    #density[i]=rhog(r[i])
    massenc[i]=mass_enc(r[i])

    print (i)
    end=time.time()
    dt=end-start
    print ("Time taken: %f s" %(dt))

fig2,ax2=plt.subplots()
ax2.semilogx(r,massenc_recovered,color='b')
ax2.semilogx(r,massenc,color='r')

plt.show()


#---------------------------------------------------------------------

























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
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker
from matplotlib.patches import BoxStyle
from scipy import integrate
from scipy import optimize
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline,interp1d
from scipy.special import erf as errorfunc
from scipy.special import gamma, factorial
from scipy.signal import argrelextrema 
from matplotlib import cm
from numpy import linalg
import time
import os
import h5py

#-----------------------------------------------------------------

#Code parameters

epsinit=1e-7
EPSABS_plummer=1e-8 #Default: 1e-8
EPSREL_plummer=1e-4 #Default: 1e-8
EPSABS=1e-7 #Default: 1e-7
EPSREL=1e-2 #Default: 1e-6
maxiter=100

#-----------------------------------------------------------------

#Functions


#Subject

def rho(r,subject_flag):
    if (subject_flag==1):
        return (3/(4*np.pi))*(1/(1+r**2)**2.5) #plummer
    elif (subject_flag==2):
        return (1/(2*np.pi))*(1/(r*(1+r)**3)) #hernquist
    elif (subject_flag==3):
        return (1/(4*np.pi))*(1/(r*(1+r)**2)) #NFW
    
def Menc(r,subject_flag):
    if (subject_flag==1):
        return r**3/(1+r**2)**1.5 #plummer
    elif (subject_flag==2):
        return r**2/(1+r)**2 #hernquist 
    elif (subject_flag==3):
        return np.log(1+r)-r/(1+r) #NFW

#.................................................................


#General

#Total


def impulse_integral(s,rp,perturber_flag):
    if (perturber_flag==1):              #Kuzmin disk
        return (1/s**2)*(1-rp/np.sqrt(rp**2+s**2))
    elif (perturber_flag==2):
        return (0.5*np.pi)/np.sqrt(rp**2+s**2) #Triaxial log

def F_integrand_disk(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*(r**2*s**2*I**2)*np.sin(theta)
    return integrand/(4*np.pi)

def F_integrand_triaxial(phi,theta,r,b,rp,q1,q2,perturber_flag,subject_flag):
    ssq=(r**2/q1**2)*np.sin(theta)**2*np.cos(phi)**2+(b-r*np.sin(theta)*np.sin(phi))**2/q2**2
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    s1sq=(r**2/q1**4)*np.sin(theta)**2*np.cos(phi)**2+(b-r*np.sin(theta)*np.sin(phi))**2/q2**4
    integrand=rho(r,subject_flag)*(r**2*s1sq*I**2)*np.sin(theta)
    return integrand/(4*np.pi)

def F_disk(b,Rcut,rp,perturber_flag,subject_flag):
    if (perturber_flag==1):
        return integrate.tplquad(F_integrand_disk,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]

def F_triaxial(b,Rcut,rp,q1,q2,perturber_flag,subject_flag):
    if (perturber_flag==2):
        return integrate.tplquad(F_integrand_triaxial,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,q1,q2,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]


#Center of mass    

def vcm_integrand_disk(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*( r**2*I*(b-r*np.sin(theta)*np.sin(phi)) )*np.sin(theta)
    return integrand

def vcm_integrand_triaxial(phi,theta,r,b,rp,q1,q2,perturber_flag,subject_flag):
    ssq=(r**2/q1**2)*np.sin(theta)**2*np.cos(phi)**2+(b-r*np.sin(theta)*np.sin(phi))**2/q2**2
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*( r**2*I*(b-r*np.sin(theta)*np.sin(phi)) )*(np.sin(theta)/q2**2)
    return integrand

def Fcm_disk(b,Rcut,rp,perturber_flag,subject_flag):
    if (perturber_flag==1):
        vcm = (1/Menc(Rcut,subject_flag))*integrate.tplquad(vcm_integrand_disk,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
        return (Menc(Rcut,subject_flag)/(4*np.pi))*vcm**2

def Fcm_triaxial(b,Rcut,rp,q1,q2,perturber_flag,subject_flag):
    if (perturber_flag==2):
        vcm = (1/Menc(Rcut,subject_flag))*integrate.tplquad(vcm_integrand_triaxial,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,q1,q2,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
        return (Menc(Rcut,subject_flag)/(4*np.pi))*vcm**2

#.................................................................



#Distant tide approximation

def Fapprox_integrand(r,subject_flag):
    return rho(r,subject_flag)*r**4

def chi_st(b,rp,perturber_flag):
    if (perturber_flag==1):           #Kuzmin disk
        Ix=1-rp/np.sqrt(rp**2+b**2)
        Iy=1-(rp*(rp**2+2*b**2))/(rp**2+b**2)**1.5
        Iz=(rp*b**2)/(rp**2+b**2)**1.5
        return Ix**2+Iy**2+Iz**2

def Ftidal(b,Rcut,rp,perturber_flag,subject_flag):
    A=(2/(3*b**4))
    f1=A*chi_st(b,rp,perturber_flag)*integrate.quad(Fapprox_integrand,epsinit,Rcut,args=(subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
    return f1
    
def Ftidal_pt(b,Rcut,subject_flag):
    A=(2/(3*b**4))
    f1=A*integrate.quad(Fapprox_integrand,epsinit,Rcut,args=(subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
    return f1






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
from scipy.special import gamma, factorial, spence
from scipy.signal import argrelextrema 
from matplotlib import cm
from numpy import linalg
import time
import os
import h5py

#-------------------------------Code parameters----------------------------

epsinit=1e-7
EPSABS_plummer=1e-8 #Default: 1e-8
EPSREL_plummer=1e-8 #Default: 1e-8
EPSABS=1e-7 #Default: 1e-7
EPSREL=1e-2 #Default: Cored subject: 1e-7, Cuspy subject: 1e-2
maxiter=100
zetamax=1e4 #Default: 1e4


#--------------------------------Profile functions---------------------------------


#Subject

def rho(r,subject_flag):
    if (subject_flag==1):
        return (3/(4*np.pi))*(1/(1+r**2)**2.5) #Plummer
    elif (subject_flag==2):
        return (1/(2*np.pi))*(1/(r*(1+r)**3)) #Hernquist
    elif (subject_flag==3):
        return (1/(4*np.pi))*(1/(r*(1+r)**2)) #NFW
    
def Menc(r,subject_flag):
    if (subject_flag==1):
        return r**3/(1+r**2)**1.5 #Plummer
    elif (subject_flag==2):
        return r**2/(1+r)**2 #Hernquist 
    elif (subject_flag==3):
        return np.log(1+r)-r/(1+r) #NFW

def Phi(r,subject_flag):
    if (subject_flag==1):
        return -1/np.sqrt(1+r**2) #Plummer
    elif (subject_flag==2):
        return -1/(1+r) #Hernquist 
    elif (subject_flag==3):
        return -np.log(1+r)/r #NFW

def sigma(r,subject_flag):
    if (subject_flag==1): #Plummer
        sigmasq=1/(6*np.sqrt(1+r**2))
        return np.sqrt(sigmasq)
    elif (subject_flag==2): #Hernquist
        sigmasq=r*(1+r)**3*np.log((1+r)/r)-(r/(12*(1+r)))*(25+52*r+42*r**2+12*r**3)
        return np.sqrt(sigmasq)
    elif (subject_flag==3): #NFW
        #Li2=-r*( 1+(10**(-0.5))*(r**(0.62/0.7)) )**(-0.7)
        Li2=spence(1+r)
        sigmasq=r*(1+r)**2*( np.pi**2-np.log(r)-1/r-1/(1+r)**2-6/(1+r)+(1+1/r**2-4/r-2/(1+r))*np.log(1+r)+3*(np.log(1+r))**2+6*Li2 )
        return np.sqrt(sigmasq)

def omega(r,subject_flag):
    return sigma(r,subject_flag)/r

def half_mass_radius_finder_nfw(r,c,x):
    return np.log(1+r)-r/(1+r)-0.5*x*(np.log(1+c)-c/(1+c))

def half_mass_radius(x,c,subject_flag):
    if (subject_flag==1):
        return 1/np.sqrt((2/x)**(2.0/3)-1) #Plummer
    elif (subject_flag==2):
        return 1/(np.sqrt(2/x)-1) #Hernquist
    elif (subject_flag==3):
        return optimize.brentq(half_mass_radius_finder_nfw,0,c,args=(c,x)) #NFW

#Perturber

def Phip(r,rp,perturber_flag):
    if (perturber_flag==1):
        return -1/r #Point mass
    if (perturber_flag==2): 
        return -1/np.sqrt(rp**2+r**2) #Plummer
    elif (perturber_flag==3):
        return -1/(rp+r) #Hernquist
    elif (perturber_flag==4):
        return -np.log(1+r/rp)/r #NFW
    elif (perturber_flag==5):
        return -1/(rp+np.sqrt(rp**2+r**2)) #Isochrone

def vesc(r,rp,perturber_flag):
    return np.sqrt(2*abs(Phip(r,rp,perturber_flag)))

def Mencp(r,rp,perturber_flag):
    if (perturber_flag==2):
        return r**3/(rp**2+r**2)**1.5 #Plummer
    elif (perturber_flag==3):
        return r**2/(rp+r)**2 #Hernquist 
    elif (perturber_flag==4):
        return np.log(1+r/rp)-r/(rp+r) #NFW
    elif (perturber_flag==5):
        return r**3/(np.sqrt(rp**2+r**2)*(rp+np.sqrt(rp**2+r**2))**2) #Isochrone

def dMencp_dlnr(r,rp,perturber_flag):
    if (perturber_flag==2):
        return (3*r**3*rp**2)/((rp**2+r**2)**2.5) #Plummer
    elif (perturber_flag==3):
        return (2*rp*r**2)/(rp+r)**3 #Hernquist
    elif (perturber_flag==4):
        return r**2/(rp+r)**2 #NFW
    elif (perturber_flag==5):
        return (r**3*(2*rp*r**2+3*rp**2*(rp+np.sqrt(rp**2+r**2))))/((rp**2+r**2)**1.5*(rp+np.sqrt(rp**2+r**2))**3) #Isochrone

#.......................................................................

#Adiabatic correction factor

def adiabatic_correction(r,b,vp,R_half_mass,Mtrunc,adcorr_flag,subject_flag):
    if (adcorr_flag==1):
        tdyn=(np.pi/np.sqrt(2))*(R_half_mass**1.5/Mtrunc**0.5)
        tau=max(b,2)/vp
        f=omega(r,subject_flag)*tau
        gamma=2.5-0.5*(1+errorfunc((tau-2.5*tdyn)/(0.7*tdyn)))
        return (1+f**2)**(-gamma)
    elif (adcorr_flag==0):
        return 1

#.............................Energy change..............................

#General formalism

#Total energy

def J_integrand_plummer(x,r,b,rp):
    C=(r**2-b**2-r**2*x**2)**2 + rp**2*(r**2+b**2-r**2*x**2)
    D=( (r**2-b**2+rp**2-r**2*x**2)**2 + 4*rp**2*b**2 )**1.5
    return C/D

def F_integrand_plummer(x,r,b,rp,subject_flag):
    return rho(r,subject_flag)*( r**2*J_integrand_plummer(x,r,b,rp) )


def impulse_integral(s,rp,perturber_flag):
    if (perturber_flag==1):              #Point mass
        return 1/s**2

    elif (perturber_flag==2):            #Plummer
        return 1/(s**2+rp**2)

    elif (perturber_flag==3):            #Hernquist
        Al=np.log((rp+np.sqrt(abs(rp**2-s**2)))/s)
        Ag=np.arctan(np.sqrt(abs((s-rp)/(s+rp))))
        Il=-1+(rp*Al)/np.sqrt(abs(rp**2-s**2))
        Ig=1-(2*rp*Ag)/np.sqrt(abs(rp**2-s**2))
        return (np.heaviside(rp-s,1)*Il+np.heaviside(s-rp,1)*Ig)/abs(rp**2-s**2)

    elif (perturber_flag==4):            #NFW
        Al=np.log((rp+np.sqrt(abs(rp**2-s**2)))/s)
        Ag=np.arctan(np.sqrt(abs((s-rp)/(s+rp))))
        return np.heaviside(rp-s,1)*((np.log(s/(2*rp))+(rp*Al)/np.sqrt(abs(rp**2-s**2)))/s**2)+np.heaviside(s-rp,1)*((np.log(s/(2*rp))+(2*rp*Ag)/np.sqrt(abs(s**2-rp**2)))/s**2)
    elif (perturber_flag==5):            #Isochrone
        return 1/s**2-((2*rp)/s**3)*np.arctan((np.sqrt(rp**2+s**2)-rp)/s)



def F_integrand(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*(r**2*s**2*I**2)*np.sin(theta)
    return integrand/(4*np.pi)

def F_singular_integrand(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*(r**2*s**2*I**2)*np.sin(theta)
    return integrand/(4*np.pi)

def lim10(theta,r,b,rp,perturber_flag,subject_flag):
    if (rp>=abs(r*np.sin(theta)-b) and rp<=(r*np.sin(theta)+b)):
        phi0=np.arcsin((r**2*np.sin(theta)**2+b**2-rp**2)/(2*b*r*np.sin(theta)))
        if (phi0>0):
            return [0,phi0*(1-epsinit)]
        else:
            return [0,(np.pi-phi0)*(1-epsinit)]
    else:
        return [epsinit,(2*np.pi)/3]

def lim20(theta,r,b,rp,perturber_flag,subject_flag):
    if (rp>=abs(r*np.sin(theta)-b) and rp<=(r*np.sin(theta)+b)):
        phi0=np.arcsin((r**2*np.sin(theta)**2+b**2-rp**2)/(2*b*r*np.sin(theta)))
        if (phi0>0):
            return [phi0*(1-epsinit),(np.pi-phi0)*(1-epsinit)]
        else:
            return [(np.pi-phi0)*(1-epsinit),(2*np.pi+phi0)*(1-epsinit)]
    else:
        return [(2*np.pi)/3,(4*np.pi)/3]

def lim30(theta,r,b,rp,perturber_flag,subject_flag):
    if (rp>=abs(r*np.sin(theta)-b) and rp<=(r*np.sin(theta)+b)):
        phi0=np.arcsin((r**2*np.sin(theta)**2+b**2-rp**2)/(2*b*r*np.sin(theta)))
        if (phi0>0):
            return [(np.pi-phi0)*(1-epsinit),2*np.pi*(1-epsinit)]
        else:
            return [(2*np.pi+phi0)*(1-epsinit),2*np.pi*(1-epsinit)]
    else:
        return [(4*np.pi/3),2*np.pi*(1-epsinit)]


def opts0(theta,r,b,rp,perturber_flag,subject_flag):
    '''if (rp>=abs(r*np.sin(theta)-b) and rp<=(r*np.sin(theta)+b)):
        phi0=np.arcsin((r**2*np.sin(theta)**2+b**2-rp**2)/(2*b*r*np.sin(theta)))
        #if (phi0>0):
        if (rp<=np.sqrt(r**2*np.sin(theta)**2+b**2)):
            return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter, 'points' : [phi0,np.pi-phi0]}
        else:
            return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter, 'points' : [np.pi-phi0,2*np.pi+phi0]}
    else:
        return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter}'''
    return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter}

def opts1(r,b,rp,perturber_flag,subject_flag):
    if (r>=(b+rp)):
        return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter, 'points' : [np.arcsin((b+rp)/r),np.arcsin(abs(b-rp)/r)]}
    elif (r<(b+rp) and r>=abs(b-rp)):
        return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter, 'points' : [np.arcsin(abs(b-rp)/r)]}
    else:
        return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter}

def opts2(b,rp,perturber_flag,subject_flag):
    return {'epsabs': EPSABS, 'epsrel': EPSREL, 'limit': maxiter, 'points' : [abs(b-rp),b+rp]}

def vx(x,y,z,b,rp,perturber_flag):
    s=np.sqrt(x**2+(b-y)**2)
    return -x*impulse_integral(s,rp,perturber_flag)

def vy(x,y,z,b,rp,perturber_flag):
    s=np.sqrt(x**2+(b-y)**2)
    return (b-y)*impulse_integral(s,rp,perturber_flag)

def vz(x,y,z,b,rp,perturber_flag):
    return 0

def F(b,Rcut,rp,perturber_flag,subject_flag):
    if (perturber_flag==2):
        return integrate.dblquad(F_integrand_plummer,epsinit,Rcut,lambda x: epsinit,lambda x: 1-epsinit,args=(b,rp,subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
    elif (perturber_flag==3 or perturber_flag==4):
        #return integrate.nquad(F_singular_integrand,[[epsinit,2*np.pi*(1-epsinit)],[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]
        return integrate.nquad(F_singular_integrand,[lim10,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(F_singular_integrand,[lim20,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(F_singular_integrand,[lim30,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]
    else:
        return integrate.tplquad(F_integrand,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]



#Center of mass energy   

def vcm_integrand_plummer(x,r,b,rp,subject_flag):
    I=1-(r**2-b**2+rp**2-r**2*x**2)/np.sqrt((r**2-b**2+rp**2-r**2*x**2)**2+4*rp**2*b**2)
    return rho(r,subject_flag)*(r**2*I)

def vcm_integrand(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*( r**2*I*(b-r*np.sin(theta)*np.sin(phi)) )*np.sin(theta)
    return integrand

def vcm_singular_integrand(phi,theta,r,b,rp,perturber_flag,subject_flag):
    ssq=r**2*np.sin(theta)**2+b**2-2*b*r*np.sin(theta)*np.sin(phi)
    s=np.sqrt(ssq)
    I=impulse_integral(s,rp,perturber_flag)
    integrand=rho(r,subject_flag)*( r**2*I*(b-r*np.sin(theta)*np.sin(phi)) )*np.sin(theta)
    return integrand

def vCM(b,Rcut,rp,perturber_flag,subject_flag):
    if (perturber_flag==2):
        vcm = ((2*np.pi)/(Menc(Rcut,subject_flag)*b))*integrate.dblquad(vcm_integrand_plummer,epsinit,Rcut,lambda x: epsinit,lambda x: 1-epsinit,args=(b,rp,subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
        return vcm
    elif (perturber_flag==3 or perturber_flag==4):
        vcm = (1/Menc(Rcut,subject_flag))*(integrate.nquad(vcm_singular_integrand,[lim10,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(vcm_singular_integrand,[lim20,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(vcm_singular_integrand,[lim30,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0])
        return vcm
    else:
        vcm = (1/Menc(Rcut,subject_flag))*integrate.tplquad(vcm_integrand,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
        return vcm

def Fcm(b,Rcut,rp,perturber_flag,subject_flag):
    if (perturber_flag==2):
        vcm = ((2*np.pi)/(Menc(Rcut,subject_flag)*b))*integrate.dblquad(vcm_integrand_plummer,epsinit,Rcut,lambda x: epsinit,lambda x: 1-epsinit,args=(b,rp,subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
        return (Menc(Rcut,subject_flag)/(4*np.pi))*vcm**2
    elif (perturber_flag==3 or perturber_flag==4):
        vcm = (1/Menc(Rcut,subject_flag))*(integrate.nquad(vcm_singular_integrand,[lim10,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(vcm_singular_integrand,[lim20,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0]+integrate.nquad(vcm_singular_integrand,[lim30,[epsinit,np.pi*(1-epsinit)],[epsinit,Rcut]],args=(b,rp,perturber_flag,subject_flag,), opts=[opts0, opts1, opts2])[0])
        return (Menc(Rcut,subject_flag)/(4*np.pi))*vcm**2
    else:
        vcm = (1/Menc(Rcut,subject_flag))*integrate.tplquad(vcm_integrand,epsinit,Rcut,lambda x: epsinit,lambda x: np.pi*(1-epsinit),lambda x,y: epsinit,lambda x,y: 2*np.pi*(1-epsinit),args=(b,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
        return (Menc(Rcut,subject_flag)/(4*np.pi))*vcm**2

#.................................................................



#Distant tide approximation

#Internal energy

def Fapprox_integrand(r,subject_flag):
    return rho(r,subject_flag)*r**4


def I0_integrand(zeta,b,rp,perturber_flag):
    zeta=np.exp(zeta)
    return Mencp(b*zeta,rp,perturber_flag)/(zeta*np.sqrt(zeta**2-1))
    
def I0(b,rp,perturber_flag):
    if (perturber_flag==1):
        return 1
    else:
        return integrate.quad(I0_integrand,np.log(1+epsinit),np.log(zetamax),args=(b,rp,perturber_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]

def I1_integrand(zeta,b,rp,perturber_flag):
    zeta=np.exp(zeta)
    return dMencp_dlnr(b*zeta,rp,perturber_flag)/(zeta*np.sqrt(zeta**2-1))
    
def I1(b,rp,perturber_flag):
    if (perturber_flag==1):
        return 0
    else:
        return integrate.quad(I1_integrand,np.log(1+epsinit),np.log(zetamax),args=(b,rp,perturber_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]

def J0_integrand(zeta,b,rp,perturber_flag):
    zeta=np.exp(zeta)
    return Mencp(b*zeta,rp,perturber_flag)/(zeta**3*np.sqrt(zeta**2-1))
    
def J0(b,rp,perturber_flag):
    if (perturber_flag==1):
        return 2/3.0
    else:
        return integrate.quad(J0_integrand,np.log(1+epsinit),np.log(zetamax),args=(b,rp,perturber_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]

def J1_integrand(zeta,b,rp,perturber_flag):
    zeta=np.exp(zeta)
    return dMencp_dlnr(b*zeta,rp,perturber_flag)/(zeta**3*np.sqrt(zeta**2-1))
    
def J1(b,rp,perturber_flag):
    if (perturber_flag==1):
        return 0
    else:    
        return integrate.quad(J1_integrand,np.log(1+epsinit),np.log(zetamax),args=(b,rp,perturber_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]

def vxtidal(x,y,z,b,rp,perturber_flag):
    return -(1/b**2)*I0(b,rp,perturber_flag)*x

def vytidal(x,y,z,b,rp,perturber_flag):
    return (1/b**2)*(3*J0(b,rp,perturber_flag)-J1(b,rp,perturber_flag)-I0(b,rp,perturber_flag))*y

def vztidal(x,y,z,b,rp,perturber_flag):
    return (1/b**2)*(2*I0(b,rp,perturber_flag)-I1(b,rp,perturber_flag)-3*J0(b,rp,perturber_flag)+J1(b,rp,perturber_flag))*z

def Ftidal(b,Rcut,rp,perturber_flag,subject_flag):
    A=(2/(3*b**4))
    chi_st=0.5*((3*J0(b,rp,perturber_flag)-J1(b,rp,perturber_flag)-I0(b,rp,perturber_flag))**2+(2*I0(b,rp,perturber_flag)-I1(b,rp,perturber_flag)-3*J0(b,rp,perturber_flag)+J1(b,rp,perturber_flag))**2+I0(b,rp,perturber_flag)**2)
    f1=A*chi_st*integrate.quad(Fapprox_integrand,epsinit,Rcut,args=(subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
    return f1
    
def Ftidal_pt(b,Rcut,subject_flag):
    A=(2/(3*b**4))
    f1=A*integrate.quad(Fapprox_integrand,epsinit,Rcut,args=(subject_flag,),epsabs=EPSABS_plummer,epsrel=EPSREL_plummer)[0]
    return f1

#.................................................................



#Head-on approximation
    
def F0_integrand_plummer(r,rp,subject_flag):
    A=(2*r**2+rp**2)/(4*r*(r**2+rp**2)**1.5)
    C=np.log((np.sqrt(r**2+rp**2)+r)/(np.sqrt(r**2+rp**2)-r))
    B=-0.5/(r**2+rp**2)
    return rho(r,subject_flag)*r**2*(A*C+B)

def Sigmas_integrand(r,R,Rcut,subject_flag):
    return (r*rho(r,subject_flag))/np.sqrt(r**2-R**2)

def Sigmas(R,Rcut,subject_flag):
    return 2*integrate.quad(Sigmas_integrand,R*(1+epsinit),Rcut,args=(R,Rcut,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]

def F0_integrand(R,rp,trunc,Sigmas_spline,Sigmastrunc_spline,perturber_flag):
    if (trunc==0):
        return 0.5*(I0(R,rp,perturber_flag)**2/R)*Sigmas_spline(R)
    else:
        return 0.5*(I0(R,rp,perturber_flag)**2/R)*Sigmastrunc_spline(R)

#.................................................................



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


#-------------------------------Code parameters----------------------------

eps=1e-5
EPSABS=1e-7 #Default: 1e-7
EPSREL=1e-2 #Default: Cored subject: 1e-7, Cuspy subject: 1e-2

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
    if (subject_flag==1):
        #return 1/(1+r**2)**(3/4.0) #Plummer
        sigmasq=1/(6*np.sqrt(1+r**2))
        return np.sqrt(sigmasq)
    elif (subject_flag==2):
        #return np.sqrt(1/(r*(1+r)**2)) #Hernquist
        sigmasq=r*(1+r)**3*np.log((1+r)/r)-(r/(12*(1+r)))*(25+52*r+42*r**2+12*r**3)
        return np.sqrt(sigmasq)
    elif (subject_flag==3):
        #return np.sqrt(np.log(1+r)/r**3-1/(r**2*(1+r))) #NFW
        Li2=-r*( 1+(10**(-0.5))*(r**(0.62/0.7)) )**(-0.7)
        sigmasq=r*(1+r)**2*(np.pi**2-np.log(r)-1/r-1/(1+r)**2-6/(1+r)+(1+1/r**2-4/r-2/(1+r))*np.log(1+r)+3*(np.log(1+r))**2+6*Li2)
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

def Phip(r,perturber_flag):
    if (perturber_flag==1): 
        return -1/(1+r) #Hernquist
    elif (perturber_flag==2):
        return -np.log(1+r)/r #NFW
    elif (perturber_flag==3):
        return -1/np.sqrt(1+r**2) #Plummer
    elif (perturber_flag==4):
        return -1/(1+np.sqrt(1+r**2)) #Isochrone

def dPhipdRp_Rp(r,rp,perturber_flag):
    if (perturber_flag==1): 
        return 1/(r*(rp+r)**2) #Hernquist
    elif (perturber_flag==2):
        return np.log(1+r/rp)/r**3-1/(r**2*(rp+r)) #NFW
    elif (perturber_flag==3):
        return 1/(rp**2+r**2)**1.5 #Plummer
    elif (perturber_flag==4):
        return (1/np.sqrt(rp**2+r**2))*(1/(rp+np.sqrt(rp**2+r**2))**2) #Isochrone

def Mencp(r,rp,perturber_flag):
    if (perturber_flag==1):
        return r**2/(rp+r)**2 #Hernquist 
    elif (perturber_flag==2):
        return np.log(1+r/rp)-r/(rp+r) #NFW
    elif (perturber_flag==3):
        return r**3/(rp**2+r**2)**1.5 #Plummer
    elif (perturber_flag==4):
        return r**3/(np.sqrt(rp**2+r**2)*(rp+np.sqrt(rp**2+r**2))**2) #Isochrone

def dMencp_dlnr(r,rp,perturber_flag):
    if (perturber_flag==1):
        return (2*rp*r**2)/(rp+r)**3 #Hernquist
    elif (perturber_flag==2):
        return r**2/(rp+r)**2 #NFW
    elif (perturber_flag==3):
        return (3*r**3*rp**2)/((rp**2+r**2)**2.5) #Plummer
    elif (perturber_flag==4):
        return (r**3*(2*rp*r**2+3*rp**2*(rp+np.sqrt(rp**2+r**2))))/((rp**2+r**2)**1.5*(rp+np.sqrt(rp**2+r**2))**3) #Isochrone

def jp(alpha,e,perturber_flag):
    if (perturber_flag==1):                                                         #Hernquist
        jsq=(1+e)**2/(alpha*(1+alpha)*(1+alpha+(1-alpha)*e)) 
        j=np.sqrt(jsq)
    elif (perturber_flag==2):
        jsq=((1+e)**2/(2*e*alpha))*(np.log(1+1/alpha)-((1-e)/(1+e))*np.log(1+((1+e)/(1-e))/alpha)) #NFW
        j=np.sqrt(jsq)
    elif (perturber_flag==3):
        jsq=((1+e)**2/(2*e*alpha))*(1/np.sqrt(alpha**2+1)-1/np.sqrt(alpha**2+((1+e)/(1-e))**2)) #Plummer
        j=np.sqrt(jsq)
    elif (perturber_flag==4):
        jsq=((1+e)**2/(2*e*alpha))*(1/(alpha+np.sqrt(alpha**2+1))-1/(alpha+np.sqrt(alpha**2+((1+e)/(1-e))**2))) #Isochrone
        j=np.sqrt(jsq)
    return j

def E0p(alpha,e,perturber_flag):
    if (perturber_flag==1):
        E0=-0.5*(alpha/(1+alpha))*(1-e)*((1+e+2*alpha)/(1+alpha+(1-alpha)*e)) #Hernquist
    elif (perturber_flag==2):
        E0=((1-e)**2/(4*e))*alpha*np.log(1+1/alpha)-((1-e**2)/(4*e))*alpha*np.log(1+((1+e)/(1-e))/alpha) #NFW
    elif (perturber_flag==3):
        E0=((1-e)**2/(4*e))*(alpha/np.sqrt(alpha**2+1))-((1+e)**2/(4*e))*(alpha/np.sqrt(alpha**2+((1+e)/(1-e))**2)) #Plummer
    elif (perturber_flag==4):
        E0=((1-e)**2/(4*e))*(alpha/(alpha+np.sqrt(alpha**2+1)))-((1+e)**2/(4*e))*(alpha/(alpha+np.sqrt(alpha**2+((1+e)/(1-e))**2))) #Isochrone
    return E0

def Rapo_finder_nfw(r,E0):
    return -np.log(1+r)/r-E0

def Rapo(E0,perturber_flag):
    if (perturber_flag==1):
        return 1-1/abs(E0) #Hernquist
    elif (perturber_flag==2):
        return optimize.brentq(Rapo_finder_nfw,0.1,10,args=(E0,)) #NFW
    elif (perturber_flag==3):
        return np.sqrt(1/E0**2-1) #Plummer
    elif (perturber_flag==4):
        return np.sqrt(1/E0**2-2/abs(E0)) #Isochrone


#-------------------------------Orbit----------------------------------


def theta_integrand1(z,alpha,e,perturber_flag):
    jsq=jp(alpha,e,perturber_flag)**2
    E0=E0p(alpha,e,perturber_flag)
    
    r1=1/(alpha-z)
    f=2*((E0-Phip(r1,perturber_flag))/jsq)-(z-alpha)**2
    return 1/np.sqrt(f)

def theta_integrand2(z,alpha,e,perturber_flag):
    jsq=jp(alpha,e,perturber_flag)**2
    E0=E0p(alpha,e,perturber_flag)
    
    r2=1/(z+alpha*((1-e)/(1+e)))
    f=2*((E0-Phip(r2,perturber_flag))/jsq)-(z+alpha*((1-e)/(1+e)))**2
    return 1/np.sqrt(f)

def thetap(r,alpha,e,perturber_flag):
    z2=1/r-alpha*((1-e)/(1+e))
    I1=integrate.quad(theta_integrand1,0,alpha*(e/(1+e)),args=(alpha,e,perturber_flag,))[0]
    I2=integrate.quad(theta_integrand2,z2,alpha*(e/(1+e)),args=(alpha,e,perturber_flag,))[0]
    return I1+I2

def Tp_integrand1(z,alpha,e,perturber_flag):
    jsq=jp(alpha,e,perturber_flag)**2
    E0=E0p(alpha,e,perturber_flag)

    r1=1/(alpha-z)
    f=2*((E0-Phip(r1,perturber_flag))/jsq)-(z-alpha)**2
    return 1/((alpha-z)**2*np.sqrt(f))

def Tp_integrand2(z,alpha,e,perturber_flag):
    jsq=jp(alpha,e,perturber_flag)**2
    E0=E0p(alpha,e,perturber_flag)

    r2=1/(z+alpha*((1-e)/(1+e)))
    f=2*((E0-Phip(r2,perturber_flag))/jsq)-(z+alpha*((1-e)/(1+e)))**2
    return 1/((z+alpha*((1-e)/(1+e)))**2*np.sqrt(f))

def Tp(alpha,e,perturber_flag):
    j=jp(alpha,e,perturber_flag)
    E0=E0p(alpha,e,perturber_flag)

    I1=integrate.quad(Tp_integrand1,0,alpha*(e/(1+e)),args=(alpha,e,perturber_flag,))[0]
    I2=integrate.quad(Tp_integrand2,0,alpha*(e/(1+e)),args=(alpha,e,perturber_flag,))[0]
    return (2/j)*(I1+I2)

def Tp0_integrand(r,E0,perturber_flag):
    f=2*(E0-Phip(r,perturber_flag))
    return 1/np.sqrt(f)

def Tp0(E0,rp,perturber_flag):
    rmax=min(Rapo(E0,perturber_flag),1/rp)
    I=integrate.quad(Tp0_integrand,0,rmax,args=(E0,perturber_flag,))[0]
    return 2*I

def alpha_finder(alpha,E0,e,perturber_flag):
    return E0p(alpha,e,perturber_flag)-E0

def ALPHA(E0,e,perturber_flag):
    if (perturber_flag==1):                                                    #Hernquist
        return (np.sqrt((1+4*E0-e**2)**2-16*E0*(1+E0)*(1-e**2))-(1+4*E0-e**2))/(4*(1+E0)*(1-e))
    else:
        return optimize.brentq(alpha_finder,1e-6,1e6,args=(E0,e,perturber_flag)) 

def orbit(alpha,e,NR,perturber_flag):
    Rp=1/alpha
    Ra=((1+e)/(1-e))*Rp
    
    R=np.linspace(Rp,Ra,NR)
    THETAp1=np.zeros(NR)
    THETAp2=np.zeros(NR)
    
    for i in range(NR):
        THETAp1[i]=thetap(R[i],alpha,e,perturber_flag)
        
    THETAp2=-THETAp1
    THETAp2=np.flip(THETAp2)
    THETAp=THETAp2[0:NR-1].tolist()+THETAp1.tolist()
    THETAp=np.array(THETAp)
    
    Rinv=np.flip(R)
    Rinv=Rinv.tolist()
    R=Rinv[0:NR-1]+R.tolist()
    R=np.array(R)
    
    return THETAp,R


#-----------------------------Energy change------------------------------------


#General formalism

#Impulse

def I1(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag):
    
    Nr=len(R)
    
    x=r*np.sin(theta)*np.cos(phi)*np.ones(Nr)
    y=r*np.sin(theta)*np.sin(phi)*np.ones(Nr)
    z=r*np.cos(theta)*np.ones(Nr)
    
    R_p=np.sqrt(x**2+(rp*R*np.cos(THETAp)-y)**2+(rp*R*np.sin(THETAp)-z)**2)
    
    I=R**2*dPhipdRp_Rp(R_p,rp,perturber_flag)
    
    return integrate.simps(I,THETAp)

def I2(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag):
    
    Nr=len(R)
    
    x=r*np.sin(theta)*np.cos(phi)*np.ones(Nr)
    y=r*np.sin(theta)*np.sin(phi)*np.ones(Nr)
    z=r*np.cos(theta)*np.ones(Nr)
    
    R_p=np.sqrt(x**2+(rp*R*np.cos(THETAp)-y)**2+(rp*R*np.sin(THETAp)-z)**2)
    
    I=R**3*rp*np.cos(THETAp)*dPhipdRp_Rp(R_p,rp,perturber_flag)
    
    return integrate.simps(I,THETAp)

def I3(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag):
    
    Nr=len(R)
    
    x=r*np.sin(theta)*np.cos(phi)*np.ones(Nr)
    y=r*np.sin(theta)*np.sin(phi)*np.ones(Nr)
    z=r*np.cos(theta)*np.ones(Nr)
    
    R_p=np.sqrt(x**2+(rp*R*np.cos(THETAp)-y)**2+(rp*R*np.sin(THETAp)-z)**2)
    
    I=R**3*rp*np.sin(THETAp)*dPhipdRp_Rp(R_p,rp,perturber_flag)
    
    return integrate.simps(I,THETAp)



def delvx(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag):
    x=r*np.sin(theta)*np.cos(phi)
    j=jp(alpha,e,perturber_flag)
    
    return -(x*I1(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag))/j

def delvy(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag):
    y=r*np.sin(theta)*np.sin(phi)
    j=jp(alpha,e,perturber_flag)

    return (I2(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag)-y*I1(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag))/j

def delvz(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag):
    z=r*np.cos(theta)
    j=jp(alpha,e,perturber_flag)

    return (I3(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag)-z*I1(r,theta,phi,alpha,e,THETAp,R,rp,perturber_flag))/j


#Center of mass

def delvcmx_integrand(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag,subject_flag):
    return r**2*rho(r,subject_flag)*np.sin(theta)*delvx(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)

def delvcmy_integrand(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag,subject_flag):
    return r**2*rho(r,subject_flag)*np.sin(theta)*delvy(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)

def delvcmz_integrand(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag,subject_flag):
    return r**2*rho(r,subject_flag)*np.sin(theta)*delvz(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)
    
def delvcm(alpha,e,THETAp,R,rmax,rp,NR,perturber_flag,subject_flag):
    
    #dvx=integrate.tplquad(delvcmx_integrand,eps,rmax,lambda x: 0,lambda x: np.pi,lambda x,y: 0,lambda x,y: 2*np.pi,args=(alpha,e,THETAp,R,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    #dvy=integrate.tplquad(delvcmy_integrand,eps,rmax,lambda x: 0,lambda x: np.pi,lambda x,y: 0,lambda x,y: 2*np.pi,args=(alpha,e,THETAp,R,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    #dvz=integrate.tplquad(delvcmz_integrand,eps,rmax,lambda x: 0,lambda x: np.pi,lambda x,y: 0,lambda x,y: 2*np.pi,args=(alpha,e,THETAp,R,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    #return dvx,dvy,dvz
    
    #dvy=integrate.tplquad(delvcmy_integrand,eps,rmax,lambda x: 0,lambda x: np.pi,lambda x,y: 0,lambda x,y: 2*np.pi,args=(alpha,e,THETAp,R,rp,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    #return dvy/Menc(rmax)
    
    j=jp(alpha,e,perturber_flag)

    return (2*j*alpha)*((1-e)/(1+e))*np.sin(THETAp[(int)(2*NR-2)])
    

def delEcm(alpha,e,THETAp,R,rmax,rp,NR,perturber_flag,subject_flag):
    I=0.5*delvcm(alpha,e,THETAp,R,rmax,rp,NR,perturber_flag,subject_flag)**2
    return I

#Internal energy

def delEint_integrand(phi,theta,r,alpha,e,THETAp,R,deltavcm,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag):

    if (adcorr_flag==1):
        #R_half_mass=1/(np.sqrt(2)-1) #hernquist
        #tdyn=(2*np.pi)/omega(R_half_mass,subject_flag)
        #tdyn=1/np.sqrt(rho(R_half_mass,subject_flag))
        tdyn=(np.pi/np.sqrt(2))*(R_half_mass**1.5/Mtrunc**0.5)
        j=jp(alpha,e,perturber_flag)
        tau=max(1/(alpha**2*j),tp)
    
        f=omega(r,subject_flag)*tau*np.sqrt(mass_ratio)*rp**1.5
        gamma=2.5-0.5*(1+errorfunc((tau*np.sqrt(mass_ratio)*rp**1.5-2.5*tdyn)/(0.7*tdyn)))
        #gamma=2.5
        #gamma=1.5
    
        ad_corr=(1+f**2)**(-gamma)
    elif (adcorr_flag==0):
        ad_corr=1

    #ad_corr=1
    
    I=r**2*rho(r,subject_flag)*np.sin(theta)*0.5*(delvx(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)**2+(delvy(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)-deltavcm)**2+delvz(phi,theta,r,alpha,e,THETAp,R,rp,perturber_flag)**2)
    return I*ad_corr

def delEint(alpha,e,THETAp,R,deltavcm,rmax,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag):
    I=integrate.tplquad(delEint_integrand,eps,rmax,lambda x: 0,lambda x: np.pi,lambda x,y: 0,lambda x,y: 2*np.pi,args=(alpha,e,THETAp,R,deltavcm,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    return I


#-----------------------------------------------------------------


#Distant tide approximation

#Internal energy

def B1(alpha,e,THETAp,R,rp,perturber_flag):
    mu=Mencp(rp*R,rp,perturber_flag)
    mu1=dMencp_dlnr(rp*R,rp,perturber_flag)
    I=((3*mu-mu1)/R)*np.cos(THETAp)**2
    return integrate.simps(I,THETAp)

def B2(alpha,e,THETAp,R,rp,perturber_flag):
    mu=Mencp(rp*R,rp,perturber_flag)
    mu1=dMencp_dlnr(rp*R,rp,perturber_flag)
    I=((3*mu-mu1)/R)*np.sin(THETAp)**2
    return integrate.simps(I,THETAp)

def B3(alpha,e,THETAp,R,rp,perturber_flag):
    mu=Mencp(rp*R,rp,perturber_flag)
    I=(mu/R)
    return integrate.simps(I,THETAp)

def chi(alpha,e,THETAp,R,rp,perturber_flag):
    jsq=jp(alpha,e,perturber_flag)**2
    return ((B1(alpha,e,THETAp,R,rp,perturber_flag)-B3(alpha,e,THETAp,R,rp,perturber_flag))**2+(B2(alpha,e,THETAp,R,rp,perturber_flag)-B3(alpha,e,THETAp,R,rp,perturber_flag))**2+B3(alpha,e,THETAp,R,rp,perturber_flag)**2)/(6*jsq)

def rsq_integrand(r,alpha,e,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag):
    if (adcorr_flag==1):
        #R_half_mass=1/(np.sqrt(2)-1) #hernquist
        #tdyn=(2*np.pi)/omega(R_half_mass,subject_flag)
        #tdyn=1/np.sqrt(rho(R_half_mass,subject_flag))
        tdyn=(np.pi/np.sqrt(2))*(R_half_mass**1.5/Mtrunc**0.5)
        j=jp(alpha,e,perturber_flag)
        tau=max(1/(alpha**2*j),tp)
    
        f=omega(r,subject_flag)*tau*np.sqrt(mass_ratio)*rp**1.5
        gamma=2.5-0.5*(1+errorfunc((tau*np.sqrt(mass_ratio)*rp**1.5-2.5*tdyn)/(0.7*tdyn)))
        #gamma=2.5
        #gamma=1.5
    
        ad_corr=(1+f**2)**(-gamma)
    elif (adcorr_flag==0):
        ad_corr=1

    #ad_corr=1

    return 4*np.pi*r**4*rho(r,subject_flag)*ad_corr

def rsq(alpha,e,Rc,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag):
    I=integrate.quad(rsq_integrand,eps,Rc,args=(alpha,e,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag,),epsabs=EPSABS,epsrel=EPSREL)[0]
    return I

def delEint_tidal(alpha,e,THETAp,R,Rc,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag):
    return chi(alpha,e,THETAp,R,rp,perturber_flag)*rsq(alpha,e,Rc,R_half_mass,Mtrunc,mass_ratio,rp,tp,adcorr_flag,perturber_flag,subject_flag)


#.................................................................






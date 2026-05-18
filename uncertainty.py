#!/usr/bin/env python
# coding: utf-8

import pint
import sys
from pint import toa
from pint import models
from pint.fitter import GLSFitter
import os
import matplotlib.pyplot as plt
import astropy.units as u
from pint.models import get_model
from pint.simulation import (
    make_fake_toas_uniform,
    make_fake_toas_fromtim,
)
from pint.residuals import Residuals, WidebandTOAResiduals
from pint.logging import setup as setup_log
from pint import dmu
from pint.config import examplefile
from pint import binaryconvert as bc
import numpy as np
import io

import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as sint
from scipy.integrate import solve_ivp

from pint.orbital import kepler
import astropy.constants as const

# Using SI units
G = 6.674299e-11
c = 299792458
day = 3600*24
yr = 365.25*day
H0 = 2.27e-18

p = sp.symbols('p')
e = sp.symbols('e')
inc = sp.symbols(r'\iota')
Omg = sp.symbols(r'\Omega')
omg = sp.symbols(r'\omega')
f = sp.symbols('f')
m1 = sp.symbols(r'm_1')
m2 = sp.symbols(r'm_2')
t = sp.symbols('t')
gamma = sp.Array([p,e,inc,Omg,omg,f])
args = sp.Array([m1,m2])
freq = sp.symbols(r'\nu')
M_tot = m1+m2


def get_uncertainties_ell1(model):
    """
    Gets the uncertainties of PINT calculated values

    Takes in binary model

    Returns uncertainties in SI units
    """

    Del_a1 = model['A1'].uncertainty.to_value('m')
    Del_pb = model['PB'].uncertainty.to_value('s')
    Del_eps1 = model['EPS1'].uncertainty.value
    Del_eps2 = model['EPS2'].uncertainty.value
    Del_sininc = model['SINI'].uncertainty.value
    Del_m2 = model['M2'].uncertainty.to_value('kg')
    Del_tasc = model['TASC'].uncertainty.to_value('s')

    return Del_a1, Del_pb, Del_eps1, Del_eps2, Del_sininc, Del_tasc, Del_m2

def partial_der_ell1():
    """
    Gives the partial derivative of the ELL1 delay
    with reference to measured PINT quantities

    returns \frac{\del t}{\del a_p}, \frac{\del t}{\del p_b}, 
            \frac{\del t}{\del eps1}, \frac{\del t}{\del eps2}, 
            \frac{\del t}{\del sin_i}, \frac{\del t}{\del t_asc},
            \frac{\del t}{\del m2}, 
    """
    
    a_p = sp.symbols(r'a_p')
    p_b = sp.symbols(r'P_b')
    sin_inc = sp.symbols(r'\sin(\iota)')
    eps1 = sp.symbols(r'\epsilon_1')
    eps2 = sp.symbols(r'\epsilon_2')
    tasc = sp.symbols(r'T_{asc}')
    f = sp.symbols('f')
    m2 = sp.symbols(r'm_2')
    
    del_r = a_p/c * (
            sp.sin(f) + .5*(eps2*sp.sin(2*f) - eps1*sp.cos(2*f)) -
            1/8*(5*eps2**2*sp.sin(f) - 3*eps2**3*sp.sin(3*f) - 2*eps1*eps2*sp.cos(f) +
                 6*eps1*eps2*sp.cos(3*f) + 3*eps1**2*sp.sin(f) + 3*eps1**2*sp.sin(3*f)) -
            1/12*(5*eps2**3*sp.sin(2*f) + 3*eps1**2*eps2*sp.sin(2*f) - 6*eps1*eps2**2*sp.cos(2*f) -
                  4*eps1**3*sp.cos(2*f) - 4*eps2**3*sp.cos(4*f) + 12*eps1**2*eps2*sp.sin(4*f) +
                  12*eps1*eps2**2*sp.cos(4*f) - 4*eps1**3*sp.cos(4*f))
            )
    
    del_s = -G*m2/c**3 * sp.log(1 - sin_inc*sp.sin(f))
    
    n_hat = sp.pi/p_b
    ddel_r_df = sp.diff(del_r, f)
    del_i = del_r * (1 - n_hat*ddel_r_df + (n_hat*ddel_r_df)**2 + 0.5*n_hat**2 * del_r * sp.diff(ddel_r_df, f))
    del_tot = del_i + del_s
    del_tot = del_tot.subs({f:2*sp.pi*(t - tasc)/p_b})
    
    partial_ell1 = sp.diff(del_tot, sp.Array([a_p, p_b, eps1, eps2, sin_inc, tasc, m2]))

    return partial_ell1

def uncertainty_from_values_ell1(model):
    """
    Gives the uncertainty of the delay from
    PINT measured quantities

    Done using simple error propogation

    Returns a function of time that calculates
    uncertainties to be added to the diagonal
    of the covariance matrix

    \del t = \sqrt{ \sum_a ( \frac{\del t}{\del a} \Delta a)^2 }
    """

    Del_a1, Del_pb, Del_eps1, Del_eps2, Del_sininc, Del_tasc, Del_m2 = get_uncertainties_ell1(model)

    partial_ell1 = partial_der_ell1()
    
    uncertainty = sp.sqrt((partial_ell1[1]*Del_pb)**2 + (partial_ell1[0]*Del_ap)**2 + (partial_ell1[2]*Del_eps1)**2 +
                      (partial_ell1[3]*Del_eps2)**2 + (partial_ell1[4]*Del_sininc)**2 + (partial_ell1[5]*Del_tasc)**2 +
                      (partial_ell1[6]*Del_m2)**2 )

    TASC = (model['TASC'].value*u.d).to_value('s')
    EPS1 = model_binary['EPS1'].value
    EPS2 = model_binary['EPS2'].value
    PB = model_binary['PB'].quantity.to_value('s')
    AP = model_binary['A1'].quantity.to_value('m')
    SINI = model_binary['SINI'].value
    M2 = model_binary['M2'].quantity.to_value('kg')
    
    unc = uncertainty.evalf(subs={tasc:TASC, eps1:EPS1, eps2:EPS2, p_b:PB, a_p:AP, sin_inc:SINI, m2:M2})
    
    sigma_unc = sp.lambdify(t, unc)

    return sigma_unc


def get_uncertainties_dd(model):
    """
    Gets the uncertainties of PINT calculated values

    Takes in binary model

    Returns uncertainties in SI units
    """

    Del_a1 = model['A1'].uncertainty.to_value('m')
    Del_pb = model['PB'].uncertainty.to_value('s')
    Del_e = model['ECC'].uncertainty.value
    Del_omg = model['OM'].uncertainty.to_value('rad')
    Del_sininc = model['SINI'].uncertainty.value
    Del_m2 = model['M2'].uncertainty.to_value('kg')
    Del_t0 = model['T0'].uncertainty.to_value('s')

    return Del_a1, Del_pb, Del_ecc, Del_omg, Del_sininc, Del_t0, Del_m2


def partial_der_dd():
    """
    Gives the partial derivative of the DD delay
    with reference to measured PINT quantities

    returns \frac{\del t}{\del a_p}, \frac{\del t}{\del p_b}, 
            \frac{\del t}{\del ecc}, \frac{\del t}{\del omg}, 
            \frac{\del t}{\del sin_i}, \frac{\del t}{\del t_0},
            \frac{\del t}{\del m2}, 
    """
    
    E = sp.symbols('E')
    p_b = sp.symbols(r'P_b')
    a_p = sp.symbols(r'a_p')
    p_b = sp.symbols(r'P_b')
    sin_inc = sp.symbols(r'\sin(\iota)')
    e = sp.symbols('e')
    omg = sp.symbols(r'\omega')
    f = sp.symbols('f')
    m2 = sp.symbols(r'm_2')
    t0 = sp.symbols('T_0')
    t = sp.symbols('t')
    
    i = sp.symbols('i')
    M = sp.symbols('M')
    
    mass_func = (4*sp.pi**2 * a_p**3 / G / p_b**2)
    m1 = (-2*mass_func*m2 + sp.sqrt(4*mass_func*m2**3*sin_inc**3))/2/mass_func
    
    dr = (p_b/sp.pi)**(-2/3) * (3*m2**2 + 6*m1*m2 + 2*m2**2)/(m1+m2)**(4/3) * (G/c**3)**(2/3)
    dth = (p_b/sp.pi)**(-2/3) * (7/2*m1**2 + 6*m1*m2 + 2*m2**2)/(m1+m2)**(4/3) * (G/c**3)**(2/3)
    g = (p_b/sp.pi)**(1/3) * e * m2*(m1+2*m2)/(m1+m2)**(4/3) * (G/c**3)**(2/3)
    
    del_r = a_p/c * (sp.sin(omg)*(sp.cos(E) - e*(1+dr)) + sp.sqrt(1-e**2*(1+dth)**2)*sp.cos(omg)*sp.sin(E))
    del_e = g*sp.sin(E)
    del_re = del_r + del_e
    del_s = -G*m2/c**3 * sp.log(1 - e*sp.cos(E) - sp.sin(inc)*(sp.sin(omg)*(sp.cos(E)-e) + sp.sqrt(1-e**2)*sp.cos(omg)*sp.sin(E)))
    ddel_re_dE = sp.diff(del_re, E)
    n_hat = sp.sqrt(G*(m1+m2)/p**3*(1-e**2))/(1-e*sp.cos(E))
    del_i = del_re*(1 - n_hat*ddel_re_dE + (n_hat*ddel_re_dE)**2 + 0.5*n_hat**2*del_re*sp.diff(ddel_re_dE, E) - .5*e*sp.sin(E)/(1-e*sp.cos(E))*n_hat**2*del_re*ddel_re_dE)
    del_tot = del_i + del_s #+ del_a
    
    # Not sure about the number of harmonics for precision
    ecc_anomaly = M + 2*sp.summation(sp.besselj(i*e, i)/i * sp.sin(i*M), (i, 1, 20))
    ecc_anom = ecc_anomaly.evalf(subs={M:sp.pi*2/p_b*(t-t0)})
    tot1 = del_tot.evalf(subs={E:ecc_anom})
    
    partial_dd = sp.diff(tot, sp.Array([a_p, p_b, e, omg, sin_inc, t0, m2]))

    return partial_dd

def uncertainty_from_values_dd(model):
    """
    Gives the uncertainty of the delay from
    PINT measured quantities

    Done using simple error propogation

    Returns a function of time that calculates
    uncertainties to be added to the diagonal
    of the covariance matrix

    \del t = \sqrt{ \sum_a ( \frac{\del t}{\del a} \Delta a)^2 }
    """

    Del_a1, Del_pb, Del_ecc, Del_omg, Del_sininc, Del_t0, Del_m2 = get_uncertainties_dd(model)

    partial_dd = partial_der_dd()
    
    uncertainty = sp.sqrt((partial_dd[1]*Del_pb)**2 + (partial_dd[0]*Del_ap)**2 + (partial_dd[2]*Del_ecc)**2 +
                      (partial_dd[3]*Del_omg)**2 + (partial_dd[4]*Del_sininc)**2 + (partial_dd[5]*Del_t0)**2 +
                      (partial_dd[6]*Del_m2)**2 )

    T0 = (model['T0'].value*u.d).to_value('s')
    ECC = model_binary['ECC'].value
    OM = model_binary['OM'].quantity.to_value('rad')
    PB = model_binary['PB'].quantity.to_value('s')
    AP = model_binary['A1'].quantity.to_value('m')
    SINI = model_binary['SINI'].value
    M2 = model_binary['M2'].quantity.to_value('kg')
    
    unc = uncertainty.evalf(subs={t0:T0, ecc:ECC, omg:OM, p_b:PB, a_p:AP, sin_inc:SINI, m2:M2})
    
    sigma_unc = sp.lambdify(t, unc)

    return sigma_unc

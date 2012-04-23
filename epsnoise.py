#!/bin/env python

from numpy import cos, sin, exp, angle, real, conj
from numpy.random import normal, random, rayleigh
from math import pi, sqrt, tanh
from scipy.special import erf
from scipy.optimize import fmin

def chi2eps(chi):
    return chi / (1 + (1 - abs(chi)**2 + 0j)**0.5)

def eps2chi(eps):
    return 2*eps/(1 + abs(eps)**2)

def addNoise(eps, nu, transform_eps=True):
    # rotate into semi-major axis frame
    e = abs(eps)
    chi = eps2chi(e)
    w = 1 # w=s^2, but we can set s=1
    z = chi*w
    S = len(e)

    sigma_n = ((1+e)*(1-e))**0.5/(nu*pi**0.5) # for flux=1
    sigma_11 = sigma_n * (3*pi/4/((1-e)**5 * (1+e)))**0.5
    sigma_12 = sigma_n * (pi/4/((1+e)**3 *(1-e)**3))**0.5
    sigma_22 = sigma_n * (3*pi/4/((1+e)**5 * (1-e)))**0.5

    dQ11 = normal(0, 1, S) # actual variance enters below
    dQ22 = normal(0, 1, S)
    dQ12 = normal(0, sigma_12, S)

    # Q11 and Q22 are correlated with rho = 0.325
    # need to take into account that variances of Q11 and Q22 change with e
    rho = 0.325
    dQ22 = (rho*sigma_22)*dQ11 + ((sigma_22**2 - (rho*sigma_22)**2)**0.5)*dQ22
    dQ11 *= sigma_11

    # now w and z become correlated
    w_ = w + dQ11 + dQ22
    z_ = z + dQ11 - dQ22
    chi1_ = z_/w_
    chi2_ = 2*dQ12/w_

    phi = angle(eps)
    chi_ = chi1_*cos(phi) - chi2_*sin(phi) + 1j*(chi1_*sin(phi) + chi2_*cos(phi))
    if transform_eps:
        return chi2eps(chi_)
    else:
        return chi_

def sampleEllipticity(size, sigma_e=0.3):
    e = rayleigh(sigma_e, size)
    # make sure no outliers are created
    # this effectively tightens the distribution
    mask = (e >= 1)
    while sum(mask) > 0:
        e[mask] = rayleigh(sigma_e, sum(mask))
        mask = (e >= 1)
    # sample unformly random orientation and create complex ellipticity
    phi = pi*random(size)
    return e*(cos(2*phi) + 1j * sin(2*phi))

def addShear(eps, g):
    return (eps + g)/(1 + eps*conj(g))


## Analytic form of Marsaglia distribution
def marsaglia_f(t, a, b):
    q = (b + a*t)/(1+t**2)**0.5
    return exp(-(a*a + b*b)/2)/(pi*(1 + t**2)) * (1 + q*exp(q**2/2)*sqrt(pi/2)*erf(sqrt(0.5)*q))

def marsaglia(t, w, z, sigma_w, sigma_z, rho):
    s = rho * sigma_z/sigma_w
    r = sigma_w/(sigma_z*sqrt(1-rho*rho))
    a = (z/sigma_z - rho*(w/sigma_w))/sqrt(1-rho*rho)
    b = w/sigma_w
    return r * marsaglia_f(r*(t-s), a, b)

def marsaglia_eps(t, eps, nu):
    e = abs(eps)
    chi = eps2chi(e)
    w = 1 # w=s^2, but we can set s=1
    z = chi*w

    sigma_n = ((1+e)*(1-e))**0.5/(nu*pi**0.5) # F = 1 here
    sigma_11 = sigma_n * sqrt(3*pi/4/((1-e)**5 * (1+e)))
    sigma_12 = sigma_n * sqrt(pi/4/((1+e)**3 *(1-e)**3))
    sigma_22 = sigma_n * sqrt(3*pi/4/((1+e)**5 * (1-e)))

    # Q11 and Q22 are correlated with rho=0.325
    rho = 0.325
    sigma_w = sqrt(sigma_11**2 + sigma_22**2 + 2*rho*sigma_11*sigma_22)
    sigma_z = sqrt(sigma_11**2 + sigma_22**2 - 2*rho*sigma_11*sigma_22)
    rho = (sigma_11**2 - sigma_22**2)/(sigma_z*sigma_w)
    return marsaglia(t, w, z, sigma_w, sigma_z, rho)

## Shear estimators from section 4
def epsilon_mean(eps, limit=1.):
    mask = (abs(eps) < limit)
    return eps[mask].mean()

def chi_responsivity(chi, limit=1.):
    mask = (abs(chi) < limit)
    return chi[mask].mean()/(2-chi[mask].std()**2)

def chi_s_mean(gamma, chi):
    g = gamma[0] + 1j*gamma[1]
    return abs(((chi - 2*g + g**2 * conj(chi))/(1+abs(g)**2 - 2*real(g*conj(chi)))).sum())

def chi_shear(chi):
    gamma = [0,0]
    gamma = fmin(chi_s_mean, gamma, args=(chi,), xtol=1e-8, disp=False)
    return gamma[0] + 1j*gamma[1]
 

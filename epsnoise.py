#!/bin/env python

from numpy import cos, sin, exp, angle, real, conj
from numpy.random import normal, random, rayleigh
from math import pi, sqrt, tanh
from scipy.special import erf
from scipy.optimize import fmin

def chi2eps(chi):
    """Calculate epsilon ellipticity from chi.
    
    Args:
        chi: a real or complex number, or a numpy array thereof.

    Returns:
        The epsilon ellipticity in the same shape as the input.
    """
    return chi / (1 + (1 - abs(chi)**2 + 0j)**0.5)

def eps2chi(eps):
    """Calculate chi ellipticity from epsilon.
    
    Args:
        eps: a real or complex number, or a numpy array thereof.

    Returns:
        The chi ellipticity in the same shape as the input.
    """
    return 2*eps/(1 + abs(eps)**2)

def addNoise(eps, nu, transform_eps=True):
    """Add noise to the ellipticity.
    
    Calculates the moments of a Gaussian-shaped galaxy with given ellipticity
    eps and their errors, assuming a correlation of the symmetric moments
    Q_11 and Q_22 of rho_n = 0.325.
    Samples from a Gaussian noise distribution, such that the significance
    equals nu, then returns the ratio of the noisy moments.

    If transform_eps = True, calls chi2eps on the result, i.e. returns
    epsilon ellipticity instead of chi.

    Args:
        eps: a real or complex number, or a numpy array thereof.
        nu: a real number or a numpy array with the same shape as eps.
        transform_eps: whether the result is return as epsilon ellipticity
            rather than chi.

    Returns:
        The noisy ellipticity measurement with the same shape as eps.
    """
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
    """Draw ellipticity samples from the Rayleigh distribution.

    Samples from the Rayleigh distribution of width sigma_e, but makes sure
    that no ellipticities with |epsilon| >= 1 are created by resampling the
    outliers.
    The orientation of the sample is uniform in the interval [0 .. pi).

    Args:
        size: A positive integer.
        sigma_e: The width parameter/mode of the Rayleigh distribution

    Returns:
        A complex numpy array of given size. 
    """
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
    """Add shear to complex ellipticity.

    Args:
        eps: The complex ellipticity epsilon as a real or complex number or 
             a numpy array thereof.
        g:   The shear as a real or complex number or a numpy array with the 
             same shape as eps.
    Returns:
        The sheared ellipticity with the same shape as eps.
    """
    return (eps + g)/(1 + eps*conj(g))


## Analytic form of Marsaglia distribution
def marsaglia_f(t, a, b):
    """Compute the distribution of t = (a+x)/(b+y) with x,y ~ N(0,1).
    
    Args:
        t: A real number or a numpy array thereof.
        a: A real number or a numpy array with the same shape as t.
        b: A real number or a numpy array with the same shape as t.

    Returns:
        The value of the distribution at each value of t. 
    """
    q = (b + a*t)/(1+t**2)**0.5
    return exp(-(a*a + b*b)/2)/(pi*(1 + t**2)) * (1 + q*exp(q**2/2)*sqrt(pi/2)*erf(sqrt(0.5)*q))

def marsaglia(t, mu_w, mu_z, sigma_w, sigma_z, rho):
    """Compute the value of the Marsaglia distribution p_M(t).
    
    Transforms the ratio of t=w/z, where w,z are drawn from a bivariate Gaussian
    distribution with variances sigma_w and sigma_z and correlation rho, into 
    the form of (a+x)/(b+y) and evaluates marsaglia_f(t, a, b).

    Args:
        t: A real number or a numpy array thereof.
        mu_w: The mean of w as a real number.
        mu_z: The mean of z as a real number.
        sigma_w: The dispersion of w as a real number.
        sigma_z: The dispersion of w as a real number.
        rho: The correlation between w and z as a real number.
            Assumed to be in the range [0 .. 1).

    Returns:
        The value of p_M(t) at each value of t.
    """
    s = rho * sigma_z/sigma_w
    r = sigma_w/(sigma_z*(1-rho*rho)**0.5)
    a = (mu_z/sigma_z - rho*(mu_w/sigma_w))/(1-rho*rho)**0.5
    b = mu_w/sigma_w
    return r * marsaglia_f(r*(t-s), a, b)

def marsaglia_eps(t, eps, nu):
    """Compute the Marsaglia distribution for the ellipticity chi.

    Calculates the moments of a Gaussian-shaped galaxy with given ellipticity
    eps and their errors, assuming a correlation of the symmetric moments
    Q_11 and Q_22 of rho_n = 0.325, such that the image has significance nu.
    Returns the Marsaglia distribution for the ratio z/w, i.e. for the
    complex ellipticity chi.

    Args:
        t: A real number or a numpy array thereof.
        eps: The ellipticity epsilon as a real or complex number.
        nu: The image signicance as a positive number.

    Returns:
        The Marsaglia distribution of chi, given the true value eps and the
        significance nu.
    """
    e = abs(eps)
    chi = eps2chi(e)
    w = 1 # w=s^2, but we can set s=1
    z = chi*w

    sigma_n = ((1+e)*(1-e))**0.5/(nu*pi**0.5) # F = 1 here
    sigma_11 = sigma_n * (3*pi/4/((1-e)**5 * (1+e)))**0.5
    sigma_12 = sigma_n * (pi/4/((1+e)**3 *(1-e)**3))**0.5
    sigma_22 = sigma_n * (3*pi/4/((1+e)**5 * (1-e)))**0.5

    # Q11 and Q22 are correlated with rho=0.325
    rho = 0.325
    sigma_w = sqrt(sigma_11**2 + sigma_22**2 + 2*rho*sigma_11*sigma_22)
    sigma_z = sqrt(sigma_11**2 + sigma_22**2 - 2*rho*sigma_11*sigma_22)
    rho = (sigma_11**2 - sigma_22**2)/(sigma_z*sigma_w)
    return marsaglia(t, w, z, sigma_w, sigma_z, rho)

## Shear estimators from section 4
def epsilon_mean(eps, limit=0.999):
    """Compute mean of the ellipticity distribution.

    Args:
        eps: A numpy array of real or complex ellipticity (epsilon) estimates.
        limit: The truncation limit, a positive number.
    
    Returns: 
        Compute the mean of the eps samples, subject to the requirement 
        |eps| < limit.
    """
    mask = (abs(eps) < limit)
    return eps[mask].mean()

def chi_responsivity(chi, limit=2.):
    """Compute shear from a sample of chi ellipticities.
    
    As chi is not an unbiased estimator of the shear, the resposivity
    correction 1 - chi.std()**2 is applied.

    Args:
        chi: A numpy array of real or complex ellipticity (chi) estimates.
        limit: The truncation limit, a positive number.
    Returns:
        The mean of the chi sampled, subject to the requirement |chi| < limit,
        corrected for the responsivity of the sample.
    """
    mask = (abs(chi) < limit)
    return chi[mask].mean()/(2-chi[mask].std()**2)

def chi_s_mean(gamma, chi):
    """Calculate the absolute value of the estimated source-plance ellipticity.
    
    Args:
        gamma: A list of the two shear components.
        chi: A numpy array of complex ellipticity (chi) estimates.

    Returns:
        The absolute value of the sum of residual source plane ellipticities.
    """
    g = gamma[0] + 1j*gamma[1]
    return abs(((chi - 2*g + g**2 * conj(chi))/(1+abs(g)**2 - 2*real(g*conj(chi)))).sum())

def chi_shear(chi):
    """Compute shear estimator that nulls the residual source-plane ellipticity.

    Runs a minimizer, initialized at (0,0), for the estimated shear, such that
    the de-lensing of the given sample of chi estimates yields an maximally
    isotropic distribution.

    Args:
        chi: A numpy array of complex ellipticity (chi) estimates.
    
    Returns:
        The shear estimate with the smallest residual source-plane ellipticity.
    """
    gamma = [0,0]
    gamma = fmin(chi_s_mean, gamma, args=(chi,), xtol=1e-8, disp=False)
    return gamma[0] + 1j*gamma[1]
 

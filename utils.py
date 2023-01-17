#!/usr/bin/env python

import numpy as Num


def deltav(mjd, ra , dec, dra , ddec, equinoxe):

    # Spherical coordinates to Cartesian coordinates position 'pos'
    pos1 = sla_dcs2c(ra, dec)
    pos2 = sla_dcs2c(ra+dra, dec+ddec)

    # Barycentric and heliocentric velocity and position of the Earth
    (pos_helio, velo_helio, pos_bary, velo_bary) = sla_epv(mjd) # Return PH, VH, PB, VB (V in AU s-1)
    (velo_bary, pos_bary, velo_helio, pos_helio) = sla_evp(mjd, equinoxe) # Return fast VB, PB, VH, PH (V in AU d-1)

    # Difference of scalar products of two 3-vectors
    return 149600e6*(sla_dvdv(velo_bary, pos2) - sla_dvdv(velo_bary, pos1))
    #return (sla_dvdv(velo_bary*86400.0, pos2) - sla_dvdv(velo_bary*86400.0, pos1))

def eccentric_anomaly(E, mean_anomaly):
    """
    Calculate the eccentric anomaly using a simplte iteration to solve
    Kepler's Equations (written by Scott Ransom)
    """
    ma = Num.fmod(mean_anomaly, 2*Num.pi)
    ma = Num.where(ma < 0.0, ma+2*Num.pi, ma)
    eccentricity = E
    ecc_anom_old = ma
    ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
    # This is a simple iteration to solve Kepler's Equation
    if (len(ecc_anom) >1):
        while (Num.maximum.reduce(Num.fabs(ecc_anom-ecc_anom_old)) > 5e-15):
            ecc_anom_old = ecc_anom
            ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)
            #print(ecc_anom_old, ecc_anom)
            
    elif(len(ecc_anom) ==1):
        while (Num.fabs(ecc_anom-ecc_anom_old) > 5e-15):
            ecc_anom_old = ecc_anom
            ecc_anom = ma + eccentricity*Num.sin(ecc_anom_old)

    return ecc_anom

def eccentric_anomaly2(e, M, tolerance=1e-14):
    """Convert mean anomaly to eccentric anomaly.
    Implemented from [A Practical Method for Solving the Kepler Equation][1]
    by Marc A. Murison from the U.S. Naval Observatory
    [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    """
    MAX_ITERATIONS = 100
    Mnorm = Num.fmod(M, 2 * Num.pi)
    E0 = M + (-1 / 2 * e ** 3 + e + (e ** 2 + 3 / 2 * Num.cos(M) * e ** 3) * Num.cos(M)) * Num.sin(M)
    dE = tolerance + 1
    count = 0
    while (Num.maximum.reduce(dE)) > tolerance:
        t1 = Num.cos(E0)
        t2 = -1 + e * t1
        t3 = Num.sin(E0)
        t4 = e * t3
        t5 = -E0 + t4 + Mnorm
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
        dE = Num.abs(E - E0)
        E0 = E
        count += 1
        if count == MAX_ITERATIONS:
            print('Did not converge after {n} iterations. (e={e!r}, M={M!r})'.format(n=MAX_ITERATIONS, e=e, M=M))
            break
    return E

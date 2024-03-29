import six
import math
import sys
from presto import psr_utils as pu
try:
    from slalib import sla_ecleq, sla_eqecl, sla_eqgal
    slalib = True
except ImportError:
    slalib = False

#
# The following are the parameters that are accepted in a
# par file when trying to determine a pulsar ephemeris.
#
#  PEPOCH   Epoch of period/frequency parameters and position (MJD)
#  F0       Pulsar rotation frequency (s-2)
#  F        Alternative for F0
#  F1       Pulsar rotation frequency derivative (s^-2)
#  F2       Pulsar rotation frequency second derivative
#  P0       Pulsar period (s).
#  P        Alternative for P0
#  P1       Pulsar period derivative (10^-15).
#  DM       Dispersion measure (pc cm^-3)
#  A1       Projected pulsar semi-major axis of 1st orbit
#  E        Eccentricity of 1st orbit
#  T0       Epoch of periastron passage of 1st orbit (MJD)
#  TASC     Epoch of ascending node passage (MJD)
#  PB       Period of 1st orbit (days)
#  OM       Longitude of periastron passage, 2st orbit (deg)
#  EPS1     First Laplace parameter [eccentricity times sin(omega)]
#  EPS2     Second Laplace parameter [eccentricity times cos(omega)]
#  EPS1DOT  Time derivative of EPS1
#  EPS2DOT  Time derivative of EPS2
#  OMDOT    Rate of periastron advance (deg/yr)
#  PBDOT    Rate of change of orbital period (10^-12)
#  XDOT     Rate of change of projected semi-major axis (-12)
#  EDOT     Rate of change of eccentricity (-12)
#
#  The following are _not_ currently implemented:
#  F3, F4, F5,...  Higher order frequency derivative terms
#  OM2DOT   Second time derivative of angle of periastron (rad/s^2)
#  X2DOT    Second time derivative of projected semi-major axis (1/s)
#

float_keys = ["F", "F0", "F1", "F2", "F3", "F4", "F5", "F6",
              "P", "P0", "P1", "P2", "P3", "P4", "P5", "P6",
              "PEPOCH", "POSEPOCH", "DM", "START", "FINISH", "NTOA",
              "TRES", "TZRMJD", "TZRFRQ", "TZRSITE", "NITS",
              "A1", "XDOT", "E", "ECC", "EDOT", "T0", "PB", "PBDOT", "OM", "OMDOT",
              "EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "TASC", "LAMBDA", "BETA",
              "RA_RAD", "DEC_RAD", "GAMMA", "SINI", "M2", "MTOT",
              "FB0", "FB1", "FB2", "ELAT", "ELONG", "LAMBDA", "BETA"]
str_keys = ["FILE", "PSR", "PSRJ", "RAJ", "DECJ", "EPHEM", "CLK", "BINARY"]
par_keys = ["PSRJ", "RAJ", "DECJ", "F0", "F1", "PEPOCH", "POSEPOCH", "DM", "BINARY", "PB", "T0", "A1", "OM", "ECC"]

class Parfile:
    def __init__(self, parfile=None):
        setattr(self,'PSRJ', "")
        setattr(self,'RAJ', "00:00:00.0")
        setattr(self,'DECJ', "00:00:00.0")
        setattr(self,'F0',0.0)
        setattr(self,'F1',0.0)
        setattr(self,'OM',0.0)
        setattr(self,'ECC',0.0)
        setattr(self,'PB',0.0)
        setattr(self,'A1',0.0)
        setattr(self,'T0',0.0)
        setattr(self,'PEPOCH',0.0)
        setattr(self,'POSEPOCH',0.0)
        setattr(self,'DM',0.0)
        setattr(self,'EPHEM','')
        setattr(self,'BINARY',"BT")

        use_eclip = False # Use ecliptic coordinates
        use_ell = False # Use elliptic coordinates

        if parfile:
            self.read(parfile)

    def __str__(self):
        out = ""
        for k, v in self.__dict__.items():
            if k[:2]!="__":
                if type(self.__dict__[k]) in six.string_types:
                    out += "%10s = '%s'\n" % (k, v)
                else:
                    out += "%10s = %-20.15g\n" % (k, v)
        return out

    def read(self, parfilenm):
        self.FILE = parfilenm
        pf = open(parfilenm)
        for line in pf.readlines():
            # Convert any 'D-' or 'D+' to 'E-' or 'E+'
            line = line.replace("D-", "E-")
            line = line.replace("D+", "E+")
            try:
                splitline = line.split()
                key = splitline[0]
                if key in str_keys:
                    setattr(self, key, splitline[1])
                elif key in float_keys:
                    try:
                        setattr(self, key, float(splitline[1]))
                    except ValueError:
                        pass
                if len(splitline)==3:  # Some parfiles don't have flags, but do have errors
                    if splitline[2] not in ['0', '1']:
                        setattr(self, key+'_ERR', float(splitline[2]))
                if len(splitline)==4:
                    setattr(self, key+'_ERR', float(splitline[3]))
            except:
                print ('')
        # Read PSR name
        if hasattr(self, 'PSR'):
            setattr(self, 'PSR', self.PSR)
        if hasattr(self, 'PSRJ'):
            setattr(self, 'PSRJ', self.PSRJ)
        # Deal with Ecliptic coords
        if (hasattr(self, 'BETA') and hasattr(self, 'LAMBDA')):
            self.use_eclip = True
            setattr(self, 'ELAT', self.BETA)
            setattr(self, 'ELONG', self.LAMBDA)
        if (slalib and hasattr(self, 'ELAT') and hasattr(self, 'ELONG')):
            self.use_eclip = True
            if hasattr(self, 'POSEPOCH'):
                epoch = self.POSEPOCH
            else:
                epoch = self.PEPOCH
            ra_rad, dec_rad = sla_ecleq(self.ELONG*pu.DEGTORAD,
                                        self.ELAT*pu.DEGTORAD, epoch)
            rstr = pu.coord_to_string(*pu.rad_to_hms(ra_rad))
            dstr = pu.coord_to_string(*pu.rad_to_dms(dec_rad))
            setattr(self, 'RAJ', rstr)
            setattr(self, 'DECJ', dstr)
        if hasattr(self, 'RAJ'):
            setattr(self, 'RA_RAD', pu.ra_to_rad(self.RAJ))
        if hasattr(self, 'DECJ'):
            setattr(self, 'DEC_RAD', pu.dec_to_rad(self.DECJ))
        # Compute the Galactic coords
        if (slalib and hasattr(self, 'RA_RAD') and hasattr(self, 'DEC_RAD')):
            l, b = sla_eqgal(self.RA_RAD, self.DEC_RAD)
            setattr(self, 'GLONG', l*pu.RADTODEG)
            setattr(self, 'GLAT', b*pu.RADTODEG)
        # Compute the Ecliptic coords
        if (slalib and hasattr(self, 'RA_RAD') and hasattr(self, 'DEC_RAD')):
            if hasattr(self, 'POSEPOCH'):
                epoch = self.POSEPOCH
            else:
                epoch = self.PEPOCH
            elon, elat = sla_eqecl(self.RA_RAD, self.DEC_RAD, epoch)
            setattr(self, 'ELONG', elon*pu.RADTODEG)
            setattr(self, 'ELAT', elat*pu.RADTODEG)
        if hasattr(self, 'P'):
            setattr(self, 'P0', self.P)
        if hasattr(self, 'P0'):
            setattr(self, 'F0', 1.0/self.P0)
        if hasattr(self, 'F0'):
            setattr(self, 'P0', 1.0/self.F0)
        if hasattr(self, 'F1'):
            setattr(self, 'P1', -self.F1/(self.F0*self.F0))
        if hasattr(self, 'FB0'):
            setattr(self, 'PB', (1.0/self.FB0)/86400.0)
        if hasattr(self, 'P0_ERR'):
            if hasattr(self, 'P1_ERR'):
                f, ferr, fd, fderr = pu.pferrs(self.P0, self.P0_ERR,
                                               self.P1, self.P1_ERR)
                setattr(self, 'F0_ERR', ferr)
                setattr(self, 'F1', fd)
                setattr(self, 'F1_ERR', fderr)
            else:
                f, fd, = pu.p_to_f(self.P0, self.P1)
                setattr(self, 'F0_ERR', self.P0_ERR/(self.P0*self.P0))
                setattr(self, 'F1', fd)
        if hasattr(self, 'F0_ERR'):
            if hasattr(self, 'F1_ERR'):
                p, perr, pd, pderr = pu.pferrs(self.F0, self.F0_ERR,
                                               self.F1, self.F1_ERR)
                setattr(self, 'P0_ERR', perr)
                setattr(self, 'P1', pd)
                setattr(self, 'P1_ERR', pderr)
            else:
                p, pd, = pu.p_to_f(self.F0, self.F1)
                setattr(self, 'P0_ERR', self.F0_ERR/(self.F0*self.F0))
                setattr(self, 'P1', pd)
        if hasattr(self, 'DM'):
            setattr(self, 'DM', self.DM)
        if hasattr(self, 'EPS1') and hasattr(self, 'EPS2'):
            self.use_ell = True
            ecc = math.sqrt(self.EPS1 * self.EPS1 + self.EPS2 * self.EPS2)
            omega = math.atan2(self.EPS1, self.EPS2)
            setattr(self, 'ECC', ecc)
            setattr(self, 'OM', omega)
        if hasattr(self, 'PB') and hasattr(self, 'A1') and not hasattr(self, 'ECC'):
            setattr(self, 'ECC', 0.0)
        pf.close()

    def write(self, parfilenm):
#    def write(self, parfilenm, p2f, param):
        out = ""
        for k in par_keys:
            if hasattr(self, k):
                v = self.__dict__[k]
                if type(self.__dict__[k]) in six.string_types:
                    out += "%s %27s\n" % (k, v)
                else:
                    out += "%-12s%20.15g\n" % (k, v)
        print (out)

        pfo = open(parfilenm,'w')
        pfo.write(out)
        pfo.close()

    def set_param(self, param, value):
        if hasattr(self, 'P0'):
            if self.P0:
                setattr(self, 'F0', 1.0/self.P0)
            else:
                setattr(self, 'F0', 0.0)
        if hasattr(self, 'P1'):
            if self.P1:
                setattr(self, 'F1', -self.P1/(self.P0*self.P0))
            else:
                setattr(self, 'F1', 0.0)
        setattr(self, param, value)



if __name__ == '__main__':
    a = Parfile(sys.argv[1])
    print (a)
    #a.write("test.par")

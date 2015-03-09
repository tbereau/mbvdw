#!/usr/bin/env python 

# Python implementation of many-body van der Waals interaction energy
#
# Tristan BEREAU (December 2013)

import sys, os
import numpy as np 
from math import *
import argparse


# Parse command-line options
parser = argparse.ArgumentParser(description=
  'Range-separated many-body dispersion energy without electron density',
  epilog='Tristan BEREAU (2014)')
parser.add_argument('--xyz', dest='xyz', type=str, required=True,
                    help='System geometry from xyz file')
parser.add_argument('--lx', dest='lx', type=float, required=False,
                    help='Box size in x direction [Angstroem]')
parser.add_argument('--ly', dest='ly', type=float, required=False,
                    help='Box size in y direction [Angstroem]')
parser.add_argument('--lz', dest='lz', type=float, required=False,
                    help='Box size in z direction [Angstroem]')
parser.add_argument('--alpha', dest='alpha', type=float, required=False,
                     help='Angle alpha for monoclinic system (degrees)')
parser.add_argument('--beta', dest='beta', type=float, required=False,
                    help='Angle beta for monoclinic system (degrees)')
parser.add_argument('--gamma', dest='gamma', type=float, required=False,
                    help='Angle gamma for monoclinic system (degrees)')

args = parser.parse_args()
if args.xyz == None:
  parser.print_help()
  exit(0)

np.set_printoptions(suppress=True)
#### Global variables ###
# Number of atoms
atNmb = 0
# Chemical element of each atom
atTyp = []
# Atomic coordinates
atCor = []
# Free-atom characteristic frequency
frqfr = []
# Free-atom frequ-dependent polarizabilities
polFreq = []
# TS polarizabilities (isotropic)
polTS = []
polTSiso = []
polMolTS = []
polMolTSiso = 0.0
# Characteristic TS frequencies
frqTS = []
frqTSiso = []
# MBD polarizabilities and C6
polMBD = []
csixMBD = []
#########################
# Chemistry data (in atomic units)
# Free-atom polarizabilities and C6 from:
# Chu and Dalgarno, J. Chem. Phys., Vol. 121, No. 9 (2004);
# von Lilienfeld and Tkatchenko, JCP, 132, 234109 (2010).
polFree = {
  'H' :  4.50,
  'He':  1.38,
  'C' : 12.00,
  'N' :  7.40,
  'O' :  5.40,
  'F' :  3.80,
  'Ne':  2.67,
  'Si': 37.00,
  'P' : 25.00,
  'S' : 19.60,
  'Cl': 15.00,
  'Ar': 11.10,
  'Br': 20.00,
  'Kr': 16.80,
  'I' : 35.00,
}
c6free  = {
  'H' :  6.50,
  'He':  1.46,
  'C' : 46.60,
  'N' : 24.20,
  'O' : 15.60,
  'F' :  9.52,
  'Ne':  6.38,
  'Si': 305.0,
  'P' : 185.0,
  'S' : 134.0,
  'Cl': 94.60,
  'Ar': 64.30,
  'Br': 162.0,
  'Kr': 130.0,
  'I' : 385.0,
}
radfree = {
  'H' : 3.10,
  'He': 2.65,
  'C' : 3.59,
  'N' : 3.34,
  'O' : 3.19,
  'F' : 3.04,
  'Ne': 2.91,
  'Si': 4.20,
  'P' : 4.01,
  'S' : 3.86,
  'Cl': 3.71,
  'Ar': 3.55,
  'Br': 3.93,
  'Kr': 3.82,
  'I' : 4.39,
}
electroneg = {
  'H' : 2.20,
  'He': 1.00,
  'C' : 2.55,
  'N' : 3.04,
  'O' : 3.44,
  'F' : 3.98,
  'Ne': 1.00,
  'Si': 1.90,
  'P' : 2.19,
  'S' : 2.58,
  'Cl': 3.16,
  'Ar': 1.00,
  'Br': 2.96,
  'Kr': 3.00,
  'I' : 2.66,
}
chemistry   = [polFree, c6free, radfree]
#### Numerical parameters ###
# Omega max in integration of alpha
omegaMax    = 100.0
# Step along omega for alpha as a function of omega
omegaStep   =   0.1
# omega threshold above which we assume alpha to be zero
alphaThres  =  0.001
# Grid around atom in volume ratio
gridMax     =   10.0 #
gridStep    =    2.0 #1.0
# Decay factor for the volume ratio weight
d_w         =    3.8
# Range-separation parameter
beta        =    1.10
mbdrad      =    1.85
### More global variables ###
omegaRng    = np.linspace(0.0,omegaMax,omegaMax/omegaStep)
# Bohr to Angstrom
b2a         = 0.529177
# (2*pi)**(3/2)
twopithreed = 15.7496099
# 3/pi
threeoverpi = 0.95492966
# 4/3
fourthird   = 1.3333333
# Damping coefficient d
dampd       = 11.0
# Damping coefficient s
damps       = 2.20
# hartree to kcal/mol
au2kcalmol  = 62751.0
# Degrees to radian
deg2rad     = 0.017453292519943295

pbc = [0,0,0]
cellh = np.zeros([3,3])
cells = np.zeros([3,3])
if args.lx == None or args.ly == None or args.lz == None:
  # No periodic boundary conditions
  args.lx = -1.0
  args.ly = -1.0
  args.lz = -1.0
  args.alpha = 90.0
  args.beta  = 90.0
  args.gamma = 90.0
else:
  # Convert to bohr
  args.lx /= b2a
  args.ly /= b2a
  args.lz /= b2a
  pbc = [args.lx, args.ly, args.lz]
  print "Box size: %7.4f*%7.4f*%7.4f Angstroem^3" \
    % (float(args.lx*b2a),float(args.ly*b2a),float(args.lz*b2a))
  if args.alpha != None:
    print "Angle alpha for monolinic system %6.2f" % float(args.alpha)
  else:
    args.alpha = 90.0
  if args.beta != None:
    print "Angle beta for monoclinic system %6.2f" % float(args.beta)
  else:
    args.beta = 90.0
  if args.gamma != None:
    print "Angle gamma for monoclinic system %6.2f" % float(args.gamma)
  else:
    args.gamma = 90.0
  if (args.alpha != 90.0 and args.beta  != 90.0) or \
     (args.alpha != 90.0 and args.gamma != 90.0) or \
     (args.beta  != 90.0 and args.gamma != 90.0):
    print "Monoclinic systems can only have one angle away from 90 degrees"
    exit(-1)
  # Build cell-parameter matrix
  cellh[0,0] = args.lx
  cellh[1,1] = args.ly
  cellh[2,2] = args.lz
  if args.alpha != 90.0:
    cellh[1,2] = cos(args.alpha*deg2rad) / \
             sqrt(1+cos(args.alpha*deg2rad)**2) * args.lz
  if args.beta  != 90.0:
    cellh[0,2] = cos(args.beta*deg2rad) / \
             sqrt(1+cos(args.beta*deg2rad)**2) * args.lz
  if args.gamma  != 90.0:
    cellh[0,1] = cos(args.gamma*deg2rad) / \
             sqrt(1+cos(args.gamma*deg2rad)**2) * args.ly
  print "Cell-parameter matrix:"
  print cellh
  cells = np.linalg.inv(cellh)
  

# Parse XYZ file
def parseXYZ():
  '''Parses xyz file and store coordinates and chemical elements'''
  global atNmb, atTyp, atCor, frqfr, polTS, polTSiso, polMolTS, \
    frqTS, polMBD, csixMBD, frqTSiso
  try:
    f = open(args.xyz,'r')
    s = f.readlines()
    f.close()
  except IOErroe, e:
    raise "I/O Error",e
  lineIndex = 0
  while lineIndex < len(s):
      if len(s[0].split()) != 0:
          atNmb = int(s[0].split()[0])
          break
      lineIndex += 1
      s = s[lineIndex:]
  # Number of atoms
  for i in range(2,len(s)):
    line = s[i].split()
    if len(s[i].split()) > 0:
        atomType = line[0]
        # Check whether we have chemistry data for this atom type
        for db in chemistry:
            if atomType not in db:
                print "Can't find atom type " + atomType + " in database."
                exit(1)
        atTyp.append(atomType)
        # Assign free-atom characteristic excitation frequency
        frqfr.append(4/3.*c6free[atomType]/polFree[atomType]**2)
        # Parse coordinates in bohr
        atCor.append(np.array([
                    float(line[1])/b2a,
                    float(line[2])/b2a,
                    float(line[3])/b2a]))
  if len(atTyp) != atNmb or len(atCor) != atNmb:
      print "Error: Inconsistent number of atoms in " + str(args.xyz)
      print "atTyp:",len(atTyp),"atNmb:",atNmb,"atCor",len(atCor)
      exit(1)
  # Initialize arrays
  size = (atNmb,len(omegaRng))
  polTS  = []
  polTSiso = []
  polMolTS = []
  frqTS = np.zeros((atNmb,3))
  frqTSiso = np.zeros(atNmb)
  polMBD = np.zeros((atNmb,3))
  csixMBD = np.zeros(atNmb)
  return 

def minImgVec(pos1, pos2):
  '''Distance between pos1 and pos2; minimum image convention 
  for periodic boundary conditions'''
  global pbc
  if (args.lx <= 0.0):
    # No periodic boundary conditions
    return pos1-pos2
  else:
    # Use cell-parameter matrix
    dist = np.dot(cells, pos1) - np.dot(cells, pos2)
    dist = dist - np.rint(dist)
    return np.dot(cellh, dist)


def atomDensFree(pos, atom):
  '''Free-atom Gaussian density'''
  ra = radfree[atTyp[atom]]
  return 1/(twopithreed*ra**3)*exp(
   -np.linalg.norm(minImgVec(pos,atCor[atom]))**2/(2.*ra**2))


def volRatio(atom, index=-1):
  '''Computes volume ratio for atom atom (ID) on a discrete 
  set of points.'''
  # Grid-point boundaries
  xmin = atCor[atom][0] - gridMax
  ymin = atCor[atom][1] - gridMax
  zmin = atCor[atom][2] - gridMax
  xmax = xmin + 2*gridMax
  ymax = ymin + 2*gridMax
  zmax = zmin + 2*gridMax
  # Initialize matrices
  num = 0.0
  den = 0.0
  x0 = xmin
  weightar = []
  while x0 < xmax:
    y0 = ymin
    while y0 < ymax:
      z0 = zmin
      while z0 < zmax:
        pos = np.array([x0,y0,z0])
        rn  = np.linalg.norm(minImgVec(pos,atCor[atom]))
        nAFree = atomDensFree(pos, atom)
        distTerm = rn**3
        if index in [0,1,2]:
          distTerm = minImgVec(pos,atCor[atom])[index]**2*rn
        fac = distTerm * nAFree
        partition = 1.
        # Voronoi
        closestAtm = atom
        shortestDis = 1000.0
        for atomj in range(atNmb):
          atomDis = np.linalg.norm(minImgVec(pos,atCor[atomj]))
          if atomDis < shortestDis:
            closestAtm = atomj
            shortestDis = atomDis
        if closestAtm == atom:
          num += fac
        else:
          num += fac*exp(-rn/(d_w*radfree[atTyp[atom]]))
        den += distTerm * atomDensFree(pos, atom)
        # Update coordinates
        z0 += gridStep
      y0 += gridStep
    x0 += gridStep
  return num/den

def characTSFreq():
  '''Computes TS characteristic frequencies for all atoms'''
  global frqTS
  for ati in range(atNmb):
    csix = c6free[atTyp[ati]]
    frqTSiso[ati] = fourthird * csix/(polTSiso[ati])**2
    for c in xrange(3):
      frqTS[ati][c] = fourthird * csix/(polTS[ati][c])**2
  return

def rvdw(atom):
  atomType = atTyp[atom]
  return (polTSiso[atom]/polFree[atomType])**(1/3.) \
    * radfree[atomType]

def mbvdw():
  '''Computes Many-body vdW energy'''
  global beta, polMBD, csixMBD
  size = (3*atNmb, 3*atNmb)
  intmat = np.zeros(size)
  Cmat   = np.zeros(size)
  for ati in range(atNmb):
    # Iterate over atoms j
    for atj in range(atNmb):
      if ati is atj:
        for ri in range(3):
          intmat[3*ati+ri,3*ati+ri] = frqTSiso[ati]**2
          Cmat[3*ati+ri,3*ati+ri]   = 1./(polTSiso[ati]*frqTS[ati][ri]**2)
      else:
        for ri in range(3):
          for rj in range(3):
            rij = minImgVec(atCor[ati],atCor[atj])
            rijn = np.linalg.norm(rij)
            # Kronecker delta between two coordinates
            delta_ab = 0
            if ri == rj:
              delta_ab = 1
            # Compute effective width sigma
            sigma = mbdrad * (rvdw(ati) + rvdw(atj))
            frac = (rijn/sigma)**beta
            expf = exp(-frac)
            intmat[3*ati+ri,3*atj+rj] = frqTSiso[ati] * frqTSiso[atj] * \
              sqrt(polTSiso[ati] * polTSiso[atj]) * (
                (-3.*rij[ri]*rij[rj] +rijn**2*delta_ab) /rijn**5 \
                * (1 - expf - beta*frac*expf) + \
                (beta*frac+1-beta)*beta*frac* \
                rij[ri]*rij[rj]/rijn**5*expf )
  # Compute eigenvalues
  eigvals,eigvecs = np.linalg.eigh(intmat)
  for i in xrange(3*atNmb):
    eigvecs[:,i] /= sqrt(np.dot(np.dot(eigvecs.transpose()[i],Cmat),
      eigvecs.transpose()[i]))
  # Group eigenvectors into components
  aggr = sum(eigvecs[:,i]*eigvecs[:,i]/eigvals[i] for i in xrange(len(eigvecs)))
  amol = np.zeros(3)
  for i in xrange(atNmb):
    for j in xrange(3):
      amol[j] += aggr[3*i+j]
  # Fractional anisotropy
  fracAniso = sqrt(0.5 * ((amol[0]-amol[1])**2 + \
    (amol[0]-amol[2])**2 + (amol[1]-amol[2])**2) \
    / (amol[0]**2 + amol[1]**2 + amol[2]**2))
  print \
    "Molecular polarizability MBD: {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}".format(
      sum(amol)/3.,amol[0],amol[1],amol[2],fracAniso)
  return .5*(sum([sqrt(eigvals[i]) for i in range(len(eigvals))]) - \
    3*sum(frqTSiso)) * au2kcalmol

def damp(dist,distf):
  '''Fermi-type damping function'''
  return 1./(1+exp(-dampd*(dist/(damps * distf)-1)))

parseXYZ()
print "Number of atoms:", atNmb
print "%4s %7s %7s %7s %7s" % ('Atom','x','y','z','F-Freq')
for at in range(atNmb):
  print "%-4s %7.4f %7.4f %7.4f %7.4f" % (atTyp[at],atCor[at][0]*b2a,
    atCor[at][1]*b2a,atCor[at][2]*b2a,frqfr[at])

# Casimir-Polder integral for AA.
print "Casimir-Polder integrals:"
for at in range(atNmb):
  print "Atom %d: %7.4f" % (int(at+1),
    3/4.*frqfr[at]*polFree[atTyp[at]]**2)


# Compute TS polarizabilities
print "\nTkatchenko-Scheffler:"
polMolTS = np.zeros(3)
for ati in range(atNmb):
  # TS polarizability is isotropic
  volratio = np.zeros(3)
  polTSat = np.zeros(3)
  for c in xrange(3):
    volratio[c] = volRatio(ati,c)
    polTSat[c] = volratio[c]*polFree[atTyp[ati]]
  polTS.append(polTSat)
  polTSiso.append(sum(polTSat)/3.)
  print " Atom %3d - volRatio: %7.4f polTS: %7.4f" % (int(ati+1),
    sum(volratio)/3.,  sum(polTSat)/3.)
  polMolTS += polTSat
  polMolTSiso += sum(polTSat)/3.
# print "Molecular polarizability TS : %7.4f %7.4f %7.4f" \
#   % (polMolTS[0],polMolTS[1],polMolTS[2])
print "Isotropic molecular polarizability: %7.4f" \
  % (polMolTSiso)


# Compute TS characteristic frequencies
# print ""
# print "TS characteristic frequencies:"
characTSFreq()
# for ati in range(atNmb):
  # print "Atom %d: %7.4f" % (int(ati+1), frqTS[ati])


# Compute many-body van der Waals
enembvdw = mbvdw()

# Compute total two-body dispersion
print ""
EsixTS = 0.0
CsixTSeffMono = np.zeros(atNmb)
for ati in range(atNmb):
  CsixTSeffMono[ati] = 3/4.*frqfr[ati]*polTSiso[ati]**2
  print "atom %3d (%s) - C6: %7.4f" % (ati+1,atTyp[ati],CsixTSeffMono[ati])
for ati in range(atNmb):
  for atj in range(ati+1,atNmb):
    dist  = np.linalg.norm(minImgVec(atCor[ati],atCor[atj]))
    distf = rvdw(ati) + rvdw(atj)
    # half from summation rule, 2* from Casimir
    EsixTS -= 2*CsixTSeffMono[ati]*CsixTSeffMono[atj] / (
      polTSiso[atj]/polTSiso[ati]*CsixTSeffMono[ati] +
      polTSiso[ati]/polTSiso[atj]*CsixTSeffMono[atj]) / \
      (dist**6) * damp(dist,distf) * au2kcalmol

print ""
print "Total two-body energy: %7.4f kcal/mol" % EsixTS
print "many-body dispersion energy: %7.4f kcal/mol" % enembvdw










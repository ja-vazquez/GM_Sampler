#
# This module implements a BAO Likelihood which is supplied as
# as chi2 table. See BAOLikelihoods.py
#

from BaseLikelihood import *
from scipy import *
from scipy.interpolate import RectBivariateSpline

class TabulatedBAOLikelihood (BaseLikelihood):
    def __init__(self,name,filename,chi2col,fid_theory, z, order_aper_apar=True):
        BaseLikelihood.__init__(self,name)
        print
        print "Loading ",filename
        inperp = 0 if order_aper_apar else 1
        inparl = 1 if order_aper_apar else 0

        data=loadtxt(filename)
        ## first let's autolearn the binning
        aperp=set()
        aparl=set()
        for line in data:
            aperp.add(line[inperp])
            aparl.add(line[inparl])
        aperp=sorted(list(aperp))
        aparl=sorted(list(aparl))
        logltab=zeros((len(aperp), len(aparl)))
        print "Aperp min,max,step,N:",aperp[0],aperp[-1], aperp[1]-aperp[0], len(aperp)
        print "Aparl min,max,step,N:",aparl[0],aparl[-1], aparl[1]-aparl[0], len(aparl)


        self.aperp=array(aperp)
        self.aparl=array(aparl)

        ## now fill in the table
        for line in data:
            ii=aperp.index(line[inperp])
            jj=aparl.index(line[inparl])
            if chi2col>0:
                chi2=line[chi2col]
                logltab[ii,jj]=-chi2/2.0
            else:
                ## col is probability, add 1e-50 to avoid taking log of zero.
                logltab[ii,jj]=log(line[chi2col*-1]+1e-50)
                
        logltab=logltab-logltab.max()
        self.loglint=RectBivariateSpline(self.aperp, self.aparl,logltab,kx=1,ky=1)
        print "Loading done"
        self.fidDaOverrd=fid_theory.DaOverrd(z)
        self.fidHIOverrd=fid_theory.HIOverrd(z)
        print "Fiducials at z=",z,":",self.fidDaOverrd,self.fidHIOverrd
        self.z=z

    def loglike_aperp_apar(self, aperp, apar):
        return self.loglint(aperp, apar)[0][0]
        
    def loglike(self):
        alphaperp=self.theory_.DaOverrd(self.z)/self.fidDaOverrd
        alphapar=self.theory_.HIOverrd(self.z)/self.fidHIOverrd
        return self.loglike_aperp_apar(alphaperp,alphapar)



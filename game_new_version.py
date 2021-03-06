
import scipy.linalg as linal
import scipy as sp
import numpy as np
import random
import sys

import numdifftools as nd

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Ellipse


## TODO Add fast/slow paramters
## TODO Put it on SimpleMC with real data
## TODO Add mpi4py
## TODO Save chains/files , maxlike, to explore with getdist


class Sample:
    def __init__(self, positions, datalike, allglikes=[]):
        self.positions = positions
        self.datalike  = datalike
        self.allglikes = allglikes

        self.weight    = 0



class Gaussian:
        #Defines the gaussian containing the information for the N samples on it
    def __init__(self, mean, cov):
        self.cov   = cov
        self.mean  = mean
        self.chol  = linal.cholesky(cov)
        self.lndet = sp.log(self.chol.diagonal()).sum()*2.0
        self.incov = linal.inv(cov)
        self.Npars = len(cov)


    def samples(self):
            #Place the position of the sample within the gaussian
            #along with its likelihood
        rsample   = sp.array([random.gauss(0., 1.) for i in sp.arange(self.Npars)])
        glikes    = -(rsample**2).sum()/2.0 - self.lndet/2.0
        positions = sp.dot(rsample, self.chol)
        positions += self.mean
        return positions, glikes


    def chisq(self, positions):
        delta = positions - self.mean
        return sp.dot(sp.dot(delta, self.incov), delta)


    def like(self, positions):
        return -self.chisq(positions)/2. - self.lndet/2.




class GMS:
    def __init__(self, likelifun, params, sigma=0.0):
        """main function, input the likelihood, set of
            parameters and their initial covariance matrix"""
        self.likeli    = likelifun
        self.params    = sp.array(params)
        self.sigma     = sp.array(sigma)
        self.Nparams   = len(params)
        self.fixcov    = False
        self.Nsamples  = 200
        self.weigmin   = 0.0
        self.blow      = 1.0  #increase the enveloping Gauss

        self.tweight   = 2.0
        self.maxGaus   = 40
        self.mineffsam = self.Nsamples*1

        self.effsample = 0.0
        self.weightmax = 0.0
        self.maxlike   = 0.0

        self.pickleBetween = False

        random.seed(100)
            #For plotting purposes
        self.plot      = False


    def run(self):
        done = False
        positions = self.params
        print 'initial position', positions

        self.GaussList    = []
        self.SamplingList = []

        self.Neffsample   = []
        self.allmean      = []
        self.allvar       = []
        self.allKL        = []

        while not done:
            sample_list, G = self.sampling(positions)

            self.GaussList.append(G)
            self.SamplingList.extend(sample_list)

            positions = self.rebuild_samples(self.SamplingList, self.GaussList)


            """Some convergence tests"""

            weights  = sp.array([sam.weight for sam in self.SamplingList])
            weights /= weights.max()
            norm     = weights.sum()
            lG   = len(self.GaussList)

            mean = self.some_stast(1)/norm
            var  = self.some_stast(2)/norm - mean**2

            self.allmean.append(mean)
            self.allvar.append(var)
            self.Neffsample.append(norm / (self.Nsamples * lG))

            #KL divergence
            weights /= norm
            KL = sp.exp(-(weights*sp.log(weights)).sum())/(self.Nsamples*lG)
            self.allKL.append(KL)

            if (KL >= 0.9):                             done = True
            #if (self.weightmax < self.tweight):         done = True
            if (len(self.GaussList) >= self.maxGaus):   done = True
            #if (self.effsample < self.mineffsam):       done = True
            if self.pickleBetween:
                if (len(self.GaussList)%100 == 1):
                    fname = '/tmp/game' +  str(len(self.GaussList)) + '.pickle'
                    cPicke.dump(self, open(fname, 'w'), -1)

            print ''
            print "#G=", len(self.GaussList), "wemax=", self.weightmax,  'maxlike=', self.maxlike
            print  "new_position=", positions, "effsamp=", self.effsample, "total samples", self.Nsamples*lG
            print ''

        if self.plot: self.plot_gaussian(self.GaussList, self.SamplingList)


    def some_stast(self, n):
        return sp.array([sam.weight*sam.positions**n for sam in self.SamplingList]).sum(axis=0)


    def glikes_to_probs(self, sam):
        if len(sam.allglikes) != len(self.GaussList):
            sys.exit("SHIT")
        probability = (sp.exp(sam.allglikes)).sum()
        return probability



    def rebuild_samples(self, SamplingList, GaussList):
        maxlike  = -1E30
        gmaxprob = -1E30
        for sam in SamplingList:
            if (sam.datalike > maxlike):
                maxlike      = sam.datalike
                maxlikesam   = sam
            sam.maxglike = self.glikes_to_probs(sam)
            if(sam.maxglike >= gmaxprob):
                gmaxprob     = sam.maxglike

            #Ask Anze why they're different
        #gmaxprob2 = self.glikes_to_probs(maxlikesam)
        #print 'double check', gmaxprob, gmaxprob2, 'AA'

        weightmax  = 0.0
        positmax   = None
        effsample  = 0
        finallist  = []

        for sam in SamplingList:
            normdlike = sp.exp(sam.datalike - maxlike)   #of data
            normgprob = sam.maxglike/gmaxprob            #of gaussian
            weight    = normdlike/normgprob              #[0, 1]
            sam.weight= weight
            effsample+= min(1.0, weight)
            if weight > weightmax:
                weightmax = weight
                positmax  = sam.positions
            if weight > self.weigmin:
                finallist.append(sam)
        if finallist != self.SamplingList:
            print 'something is fishy'
            self.SamplingList =finallist

        self.effsample = effsample
        self.weightmax = weightmax
        self.maxlike   = maxlike

        return positmax



    def getcov(self, params):
        Nparams = self.Nparams

        if (self.fixcov):
                #assuming the covariance is fix through the sampling process
            cov = sp.zeros((Nparams, Nparams))
            for i in sp.arange(Nparams):
                cov[i, i] = self.sigma[i]**2
                #Define a Gaussian with its samples inside
            print cov
            G = Gaussian(params, cov)
            return G
        else:
            #in the previous version, we made the Hessian by hand
            icov = -nd.Hessian(self.likeli)(params)
            while True:
                #print "Try cholesky"
                #in the previous version, first we regularized and then try cholesky
                try:
                    ch =linal.cholesky(icov)
                    break
                except:
                    print 'Regularize, removing negative eigenvectors'
                    evl, evc = linal.eig(icov)
                    evl      = sp.real(evl)
                    for i, _ in enumerate(evl):
                        if (evl[i] < 0):
                            evl[i] = sp.dot(evc[i]**2, 1./self.sigma**2)
                    icov = sp.dot(sp.dot(evc, sp.diag(evl)), evc.T)
                    #for i in sp.arange(Nparams):
                    #    icov[i, i] += 1./self.sigma[i]**2
                    pass

            cov = linal.inv(icov)
            G   = Gaussian(params, cov*self.blow)
            return G



    def sampling(self, params):
            #Get local covariance matrix
        G = self.getcov(params)

            #First update the old samples, the first step is empty
        for _ , sam in enumerate(self.SamplingList):
            sam.allglikes.append(G.like(sam.positions))

        gsamlist   = []
        for _ in sp.arange(self.Nsamples):
            positions, glikes = G.samples()
            datalike          = self.likeli(positions) #modified
                 #list of likelihood of this point
            allglikes         = [g.like(positions) for g in self.GaussList] + [glikes]
            gsamlist.append(Sample(positions, datalike, allglikes))

        return gsamlist, G


#---- Plotting tests ----------------#

    def plot_ellipse(self, G, ax, i=0, elipse=True):

        mn, cov     = G.mean, G.cov
        val, eigvec = linal.eig(cov)
        vec         = eigvec.T

        if i == 0:
            cols   = ['red']
            sigmas = [1]
        else:
            cols   = ['orange', 'green', 'blue', 'purple', 'red']*int(self.maxGaus/5.)
            sigmas = [2.3 , 5.99 , 11.83]

        vec[0] *= sp.sqrt(sigmas[0]*sp.real(val[0]))
        vec[1] *= sp.sqrt(sigmas[0]*sp.real(val[1]))
        print vec[0], vec[1], 'A', 'B'

        ax.plot([mn[0]-vec[0][0], mn[0]+vec[0][0]],
                    [mn[1]-vec[0][1], mn[1]+vec[0][1]], color=cols[i])
        ax.plot([mn[0]-vec[1][0], mn[0]+vec[1][0]],
                    [mn[1]-vec[1][1], mn[1]+vec[1][1]], color=cols[i])
        ax.plot(mn[0], mn[1], 'bo', markersize=5, color='red')

        if elipse:
            def eigsorted(cov):
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                return vals[order], vecs[:,order]

            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))


            for sigs in sigmas:
                w, h = 2  * np.sqrt(vals) * np.sqrt(sigs)
                ell = Ellipse(xy=(mn[0], mn[1]),  width = w, height = h,
                                angle=theta, color=cols[i], lw=1, alpha=0.75)
                ell.set_facecolor('none')
                ax.add_artist(ell)


    def plot_gaussian(self, GaussianList, SampligList):
        fig = plt.figure(figsize=(7,6))
        ax  = fig.add_subplot(111, aspect='equal')
        plt.grid(linewidth=1, alpha=0.3)

        cmap = plt.cm.hsv
        norm = matplotlib.colors.Normalize(vmin=0, vmax=8.)

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in xrange(0, len(l), n):
                yield l[i:i + n]

        slist = chunks(SampligList, len(GaussianList))
        for i, sl in enumerate(slist):
            positions = [s.positions for s in sl]
            colors    = 2*i+sp.rand(50)
            ax.scatter(zip(*positions)[0], zip(*positions)[1], c=cmap(norm(colors)), s =2, alpha=0.75)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm)

        print "---now plotting --"
        for i, G in enumerate(GaussianList):
            if (len(GaussianList) %5 == 0):
                self.plot_ellipse(G, ax, i)
            else:
                self.plot_ellipse(G, ax, 0)
        plt.savefig('GMS.pdf')


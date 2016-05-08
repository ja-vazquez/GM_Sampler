from scipy import *
import random
import scipy.linalg as la
import cPickle
import pylab

class Sample:
    def __init__ (self,pars, like, glikes):
        self.pars=pars
        self.like=like
        self.glikes=glikes
        
class Gaussian:
    def __init__(self,mean,cov, fastpars=None):
        self.cov=cov
        self.mean=mean
        self.chol=la.cholesky(cov)
        self.lndet=log(self.chol.diagonal()).sum()*2.0
        self.icov=la.inv(cov)
        self.N=len(cov)
        if (fastpars!=None):
            Nf=len(fastpars)
            meanf=None
            covif=zeros((Nf,Nf))
            covf=zeros((Nf,Nf))
            N=len(self.cov)
            Ns=N-Nf
            Css=zeros((Ns,Ns))
            Cfs=zeros((Nf,Ns))
            slowpars=range(N)
            for i in fastpars:
                slowpars.pop(slowpars.index(i))
            
            for i,ip in enumerate(fastpars):
                for j,jp in enumerate(fastpars):
                    covif[i,j]=self.icov[ip,jp]
                    covf[i,j]=self.cov[ip,jp]

            covf=la.inv(covif)
            
            ## yes cov here, icov above, see
            for i,ip in enumerate(slowpars):
                for j,jp in enumerate(slowpars):
                    Css[i,j]=self.cov[ip,jp]
                    
            for i,ip in enumerate(fastpars):
                for j,jp in enumerate(slowpars):
                    Cfs[i,j]=self.cov[ip,jp]
                    
            self.SubMatMu=dot(Cfs,la.inv(Css))
            tmp=la.cholesky(Css)
            tmpi=la.inv(tmp)
            ## is this stabler?
            self.SubMatMu=dot(dot(Cfs,tmpi),transpose(tmpi))
           

            self.Fast=Gaussian(None,covf,None)
            self.fastpars=fastpars
            self.slowpars=slowpars
            self.Ns=Ns
            
    def sample(self):
        da=array([random.gauss(0.,1.) for x in range(self.N)])
        glike = -(da**2).sum()/2.0-self.lndet/2.0
        sa=dot(da,self.chol)
        if (self.mean!=None):
            sa+=self.mean
        return sa,glike

    def sample_fast(self, slowsamp):
        ## here we replace slowsamps relevant pars
        sa,glike=self.Fast.sample()
        outsamp=slowsamp*1.0

        ## now get the mean
        mn=zeros(self.Ns)
        for i,ip in enumerate(self.slowpars):
            mn[i]=slowsamp[ip]-self.mean[ip]


        mn=dot(self.SubMatMu,mn)

        for i,ip in enumerate(self.fastpars):
            outsamp[ip]=self.mean[ip]+mn[i]+sa[i]
        ## but let's just calculate like by bruteforce
        glike=self.like(outsamp)
        return outsamp, glike
    
    def chi2(self,vec):
        if mean!=None:
            delta=vec-self.mean
        else:
            delta=vec
        return dot(dot(delta,self.icov),delta)
        
    def like(self,vec):
        return -self.chi2(vec)/2-self.lndet/2.0
        

class Game:
    def __init__ (self, likefuncmany, par0, sigreg=0.0):
        #random.seed(10)
        self.like=likefuncmany ## returns log like
        self.sigreg=array(sigreg)
        self.sigreg2=self.sigreg**2
        self.N=len(par0)
        self.N1=1000 ## how many samples for each Gaussian
        self.N1f=4 ## subsample fast how much
        self.blow=2.0 ## factor by which to increase the enveloping Gauss
        self.wemin=0.00  ### outputs weights with this shit
        self.mineffsamp=5000 ### minimum number effective samples that we require
        self.fixedcov=False
        self.fixedcovuse=None
        self.toexplore=array(par0)
        self.maxiter=30
        self.fastpars=None
        self.priorlow=None
        self.priorhigh=None
        self.pickleBetween=False

    def run(self):
        
        if self.fastpars==None:
            self.N1f=0
        done=False
        toexplore=self.toexplore
        badlist=[]
        self.Gausses=[]
        self.SamList=[]
        while not done:
            sample_list, G=self.isample (toexplore)
            self.Gausses.append(G)
            self.SamList+=sample_list

            toexplore=self.rebuild_samples(self.SamList, self.Gausses)
            
            if self.pickleBetween:
                if (len(self.Gausses)%100==1):
                    fname='/tmp/game'+str(len(self.Gausses))+'.pickle'
                    cPickle.dump(self,open(fname,'w'),-1)

            if (len(self.Gausses)>=self.maxiter):
                print "Max iter exceeded"
                done=True
            if (self.effsamp>self.mineffsamp):
                done=True

    def gausses_eval(self,sam):
        if len(sam.glikes)!=len(self.Gausses):
            stop("SHIT")
        probi=(exp(array(sam.glikes))).sum()
        return probi

    def rebuild_samples(self, SamList,Gausses):
        maxlike=-1e30
        gmaxlike=-1e30
        for sa in SamList:
            if (sa.like>maxlike):
                maxlike=sa.like
                maxlikesa=sa
            sa.glike=self.gausses_eval(sa) 
            if (sa.glike>gmaxlike):
                gmaxlike=sa.glike
                
        gmaxlike2=self.gausses_eval(maxlikesa)
        #gmaxlike=gmaxlike2
        wemax=0.0
        flist=[]
        wemax=0.0
        parmaxw=None
        effsamp=0
        for sa in SamList:
            rellike=exp(sa.like-maxlike)
            glike=sa.glike/gmaxlike
            we=rellike/glike
            sa.we=we
            if we>wemax:
                wemax=we
                parmaxw=sa.pars
            if we>self.wemin:
                flist.append(sa)

        #The highest weight counts one, others less
        wei=array([sa.we for sa in SamList])
        wei/=wei.max()
        effsamp=wei.sum()

        self.sample_list=flist
        print "#G=",len(Gausses), "maxlike=",maxlike,"wemax=",wemax,"effsamp=",effsamp
        self.effsamp=effsamp
        self.wemax=wemax
        return parmaxw

                        
    def getcov(self, around):
        N=self.N

        if (self.fixedcov):
            if (self.fixedcovuse!=None):
                G=Gaussian(around,self.fixedcovuse,self.fastpars) 
                return G
            else:
                cov=zeros((N,N))
                for i in range(N):
                    cov[i,i]=self.sigreg2[i]
                #print cov
                G=Gaussian(around,cov,self.fastpars)    
                return G

        icov=zeros((N,N))
        delta=self.sigreg/20.0
        toget=[]
        toget.append(around)
        
        ### This is a kinda ugly hack
        ### We repeat the exactly the same loop twice.
        ### first populating where to evaluate like 
        ### and the popping hoping for perfect sync
        fastpars=self.fastpars
        if fastpars==None:
            fastpars=[]


        for i in range(N):
            parspi=around*1.0
            parsmi=around*1.0
            parspi[i]+=delta[i]
            parsmi[i]-=delta[i]
            for j in range(i,N):
                if (i==j):
                    toget.append(parspi)
                    toget.append(parsmi)
                else:
                    #if (i not in fastpars) and (j not in fastpars):
                    parspp=parspi*1.0
                    parspm=parspi*1.0
                    parsmp=parsmi*1.0
                    parsmm=parsmi*1.0
                    parspp[j]+=delta[j]
                    parspm[j]-=delta[j]
                    parsmp[j]+=delta[j]
                    parsmm[j]-=delta[j]
                    toget.append(parspp)
                    toget.append(parsmm)
                    toget.append(parspm)
                    toget.append(parsmp)

        print "Doing covariance matrix",len(toget), N
        likes=self.like(toget)

        like0=likes.pop(0)
        for i in range(N):
            for j in range(i,N):
                if (i==j):
                    der=(likes.pop(0)+likes.pop(0)-2*like0)/(delta[i]**2)
                else:
                    #if (i not in fastpars) and (j not in fastpars):
                    der=(likes.pop(0)+likes.pop(0)-likes.pop(0)-likes.pop(0))/(4*delta[i]*delta[j])
                    #else:
                    #    der=0
                icov[i,j]=-der
                icov[j,i]=-der


        print "Checking diagonal derivatives,",
        fx=0
        for i in range(N):
            if icov[i,i]<0:
                fx+=1
                icov[i,:]=0
                icov[:,i]=0
                icov[i,i]=1/self.sigreg2[i]
        print "fixed:",fx

        print "Trying cholesky:",
        try:
            ch=la.cholesky(icov)
            print "OK"
        except:
            print "Failed, removing negative eigenvectors"
            evl,evc=la.eig(icov)
            evl=real(evl)
            for i in range(len(evl)):
                if (evl[i]<=0):
                    ## these guys we set to sensible values
                    evl[i]=dot(evc[i]**2,1/self.sigreg2)
            #evl=abs(evl) ## when nothing better?
            icov=dot(dot(evc, diag(evl)),transpose(evc))
             
        cov=la.inv(icov)
        print "Checking directions that are too small/big,",
        fx=0
        fxs=0
        for i in range(N):
            if cov[i,i]>self.sigreg2[i]:
                fx+=1
                fact=sqrt(self.sigreg[i]**2/cov[i,i])
                cov[i,:]*=fact
                cov[:,i]*=fact
            elif (cov[i,i]<self.sigreg2[i]*0.01):
                fxs+=1
                fact=sqrt(self.sigreg[i]**2*0.01/cov[i,i])
                cov[i,:]*=fact
                cov[:,i]*=fact


        print "fixed:",fxs,fx
        #print cov
        G=Gaussian(around,self.blow*cov, self.fastpars)    
        return G


    def isample (self, zeropar):
        


        ## Get local covariance matrix
        G=self.getcov(zeropar)
        
        ### first update the old samples
        for i,s in enumerate(self.SamList):
            self.SamList[i].glikes.append(G.like(s.pars))
            
        slist=[]
        lmany=[]
        fastsub=[False]+self.N1f*[True]
        
        ## now sample around this 
        for i in range(self.N1):
            for fast in fastsub:
                if fast:
                    par,glike=G.sample_fast(slowsamp)
                else:
                    par,glike=G.sample()
                    slowsamp=par
                if self.priorlow!=None:
                    if ((par<self.priorlow).any()):
                        continue
                if self.priorhigh!=None:
                    if ((par>self.priorhigh).any()):
                        continue
                glikel=[g.like(par) for g in self.Gausses] + [glike]
                lmany.append(par)
                like=None
                slist.append(Sample(par,like, glikel))
        likes=self.like(lmany)
        for like,sa in zip(likes,slist):
            sa.like=like

        return slist,G


def plotel(G,i=0, j=1,fmt='r-',times=1):
    cov=array([[G.cov[i,i], G.cov[i,j]],[G.cov[j,i], G.cov[j,j]]])
    if i==j:
        cov[1,0]=0
        cov[0,1]=0
    print cov
    val,vec=la.eig(cov)
    vec=vec.T

    vec[0]*=sqrt(real(val[0]))*times
    vec[1]*=sqrt(real(val[1]))*times
    print G.mean[i],G.mean[j]
    pylab.plot(G.mean[i],G.mean[j],'bo')
    pylab.plot([G.mean[i]-vec[0][0],G.mean[i]+vec[0][0]],
               [G.mean[j]-vec[0][1],G.mean[j]+vec[0][1]],fmt)

    pylab.plot([G.mean[i]-vec[1][0],G.mean[i]+vec[1][0]],
               [G.mean[j]-vec[1][1],G.mean[j]+vec[1][1]],fmt)


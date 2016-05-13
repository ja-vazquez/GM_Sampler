#!/usr/bin/env python
import game as gs
import scipy as sp
import pylab, sys
import time
import pandas as pd
import matplotlib.pylab as plt 
from mpi4py import MPI
from matplotlib import cm
#import scipy.linalg as linalg

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


#sp.random.seed(100)

def likemany(x):
    return map(like,x)


if sys.argv[1]=='gauss':
    def like(x):
        return -(x[0]**2+ x[1]**2/1.5 + 0.0*x[0]*x[1])/2.0
    ga=gs.Game(likemany,[0.5,-1.5],[0.7, 0.7])
    ga.N1=300
    ga.maxiter= 50
    ga.N1f=0
    #ga.fastpars=[1]
    ga.blow=1.3
    ga.mineffsamp=80000
    ga.fixedcov= False
    ga.verbose = False
    sname='gauss.pdf'
    initime=time.time()
    ga.run()
    print 'total time =',time.time()-initime

elif sys.argv[1]=='tgauss':
    def like(x):
        if x[0]>3:
            return -((x[0]-5)**2+ x[1]**2/1.5)  
        if x[0]<-3:
            return -((x[0]+5)**2+ x[1]**2/1.5)
        else:
            return -(x[0]**2+ x[1]**2/1.5 + 0.0*x[0]*x[1])/2.0
    ga=gs.Game(likemany,[0.5,-1.5],[1., 1.])
    ga.N1=100
    ga.maxiter= 90
    ga.N1f=0
    ga.fastpars=[1]
    ga.blow=1.
    ga.mineffsamp=100000
    ga.fixedcov= False
    ga.verbose = False
    sname='tgauss.pdf'
    initime=time.time()
    ga.run()
    print 'total time =',time.time()-initime
    

elif sys.argv[1]=='ring':
    def like(x):
        r2=x[0]**2+x[1]**2
        return -(r2-4.0)**2/(2*0.5**2)
    ga=gs.Game(likemany,[3.5,0.0],[0.5,0.5])
    ga.blow=2.0
    ga.N1=100
    ga.maxiter=70
    ga.fixedcov= False
    ga.verbose =False
    sname='ring.pdf'
    initime=time.time()
    ga.run()
    print 'total time =',time.time()-initime

elif sys.argv[1]=='box':
    def like(x):
        if (abs(x[0])>2) or (abs(x[1])>2):
            return -30
        else:
            return 0
    ga=gs.Game(likemany,[2.5,0.0],[.6,.6])
    #ga.priorlow=sp.array([-2.5,-2.5])
    #ga.priorhigh=sp.array([+2.5,+2.5])
    ga.N1=200
    ga.mineffsamp=60000
    ga.maxiter=40
    ga.fixedcov= False
    ga.verbose =False
    #ga.fixedcovuse= sp.array([[2,0],[0,2]])
    ga.run()
    sname='box.pdf'
    

else:
    sp.stop ("define")

if myrank ==0:
 if False:
     fig, ax = plt.subplots()
     pd.Series(ga.all_KL_div).plot()
     plt.title('Kullback-Leibler (KL) Divergence')
     ax.set_ylabel('KL')
     ax.set_xlabel('# of Gauss')
     pylab.savefig('KL.pdf')
     plt.show()
 if False:
     fig, ax = plt.subplots()
     pd.Series(ga.Neffsample).plot()
     plt.title('$N_{eff}$')
     ax.set_ylabel('Neff')
     ax.set_xlabel('# of Gauss')     
     pylab.savefig('conv.pdf')
     plt.show()
 if True:      
     all_mean =pd.DataFrame(ga.all_mean, columns =['x0', 'x1'])
     all_mean['num'] = sp.arange(ga.maxiter)
     all_var =pd.DataFrame(ga.all_var, columns =['x0', 'x1'])     
     
     cmap = cm.get_cmap('cool')
     plt.figure(figsize=(15, 6))
     plt.title('Each G')
     ax1 = pylab.subplot(1,2,1)
     sc =ax1.scatter(all_mean['x0'], all_mean['x1'], c=all_mean['num'], s=100, cmap=cmap) 
     cbar = plt.colorbar(sc) 
     cbar.set_label('# of Gauss')
     ax1.set_title('mean')
     plt.grid()
         
     ax2 = pylab.subplot(1,2,2)
     sc =ax2.scatter(all_var['x0'], all_var['x1'], c=all_mean['num'], s=100, cmap=cmap)     
     cbar = plt.colorbar(sc)
     cbar.set_label('# of Gauss')
     ax2.set_title('variance')
     plt.grid()
     
     plt.savefig('all_means.pdf')
     #plt.show()
##---- now we plot --------------------------------------------
xx= sp.array([sa.pars[0] for sa in ga.sample_list])
yy= sp.array([sa.pars[1] for sa in ga.sample_list])
ww= sp.array([sa.we      for sa in ga.sample_list])



Np   = 100
cmin = -7.
cmax = 7.
cstep= (cmax-cmin)/(1.0*Np)

sums  = sp.zeros((Np,Np))
wsums = sp.zeros((Np,Np))
trvals= sp.zeros((Np,Np))

    #fill the grid
for x, y, w in zip(xx, yy, ww):
    if (x<cmin) or (x>cmax) or (y<cmin) or (y>cmax):
        continue
    ix= int((x-cmin)/cstep)
    iy= int((y-cmin)/cstep)
    sums[iy,ix] += 1.0
    wsums[iy,ix]+= w

for i in range(Np):
    x= cmin+(i+0.5)*cstep
    for j in range(Np):
        y= cmin+(j+0.5)*cstep
        trvals[j,i]= sp.exp(like([x,y]))


trvalsa= trvals/trvals.sum()
wsumsa = wsums/wsums.sum()
diffp  = wsumsa-trvalsa
vmax   = trvalsa.max()*1.1


extent=[cmin,cmax,cmin,cmax]
plt.figure(figsize=(15, 12))
pylab.subplot(2,2,1)
pylab.imshow(trvalsa, interpolation='nearest', origin='lower left',extent=extent, vmin=0, vmax=vmax)
pylab.colorbar()

pylab.subplot(2,2,2)
pylab.imshow(sums, interpolation='nearest', origin='lower left', extent=extent)
pylab.colorbar()
for i,G in enumerate(ga.Gausses):    
    if i==0:
        gs.plotel(G,fmt='r-')
    else:
        gs.plotel(G, verbose=False)
pylab.xlim(cmin,cmax)
pylab.ylim(cmin,cmax)

pylab.subplot(2,2,3)
pylab.imshow(wsumsa, interpolation='nearest', origin='lower left',extent=extent, vmin=0, vmax=vmax)
pylab.colorbar()

pylab.subplot(2,2,4)
pylab.imshow(diffp, interpolation='nearest', origin='lower left',extent=extent)
pylab.colorbar()

#print trvalsa.max(), wsumsa.max(),diffp.max()
#print trvalsa.min(), wsumsa.min(),diffp.min()


mx= (xx*ww).sum()/(ww.sum())
vx= sp.sqrt((xx**2*ww).sum()/(ww.sum())-mx*mx)
my= (yy*ww).sum()/(ww.sum())
vy= sp.sqrt((yy**2*ww).sum()/(ww.sum())-my*my)
rr= xx**2+yy**2
mr= (rr*ww).sum()/(ww.sum())
vr= sp.sqrt((rr**2*ww).sum()/(ww.sum())-mr*mr)

#print 'xmean,xvar=',mx,vx
#print 'ymean,yvar=',my,vy
#print 'rmean,rvar=',mr,vr


pylab.savefig(sname)
#pylab.show()






#!/usr/bin/env python
import game as gs
import scipy as sp
import pylab, sys

#import scipy.linalg as linalg

sp.random.seed(100)

def likemany(x):
    return map(like,x)


if sys.argv[1]=='gauss':
    def like(x):
        return -((x[0])**2+(x[1])**2/1.0-1.5*x[0]*x[1])/2.0
    ga=gs.Game(likemany,[0.5,-1.5],[1.0,1.0])
    ga.N1=10
    ga.maxiter= 3
    ga.N1f=0
    ga.fastpars=[1]
    ga.blow=1.3
    ga.mineffsamp=8000
    ga.fixedcov= False
    ga.verbose = False
    sname='gauss.pdf'
    ga.run()


elif sys.argv[1]=='ring':
    def like(x):
        r2=x[0]**2+x[1]**2
        return -(r2-4.0)**2/(2*0.5**2)
    ga=gs.Game(likemany,[3.5,0.0],[0.4,0.4])
    ga.blow=2.0
    ga.N1=50
    ga.maxiter=30
    ga.fixedcov= False
    ga.verbose =False
    sname='ring.pdf'
    ga.run()


elif sys.argv[1]=='box':
    def like(x):
        if (abs(x[0])>2) or (abs(x[1])>2):
            return -30
        else:
            return 0
    ga=gs.Game(likemany,[2.5,0.0],[1.0,1.0])
    #ga.priorlow=array([-2.5,-2.5])
    #ga.priorhigh=array([+2.5,+2.5])
    ga.N1=50
    ga.mineffsamp=6000
    ga.maxiter=20
    ga.fixedcov= False
    ga.fixedcovuse= sp.array([[2,0],[0,2]])
    ga.run()
    sname='box.pdf'
else:
    sp.stop ("define")




##---- now we plot --------------------------------------------
xx= sp.array([sa.pars[0] for sa in ga.sample_list])
yy= sp.array([sa.pars[1] for sa in ga.sample_list])
ww= sp.array([sa.we      for sa in ga.sample_list])


Np   = 100
cmin = -5.
cmax = 5.
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
vmax   = trvalsa.max()*1.


extent=[cmin,cmax,cmin,cmax]

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

print trvalsa.max(), wsumsa.max(),diffp.max()
print trvalsa.min(), wsumsa.min(),diffp.min()


mx= (xx*ww).sum()/(ww.sum())
vx= sp.sqrt((xx**2*ww).sum()/(ww.sum())-mx*mx)
my= (yy*ww).sum()/(ww.sum())
vy= sp.sqrt((yy**2*ww).sum()/(ww.sum())-my*my)
rr= xx**2+yy**2
mr= (rr*ww).sum()/(ww.sum())
vr= sp.sqrt((rr**2*ww).sum()/(ww.sum())-mr*mr)

print 'xmean,xvar=',mx,vx
print 'ymean,yvar=',my,vy
print 'rmean,rvar=',mr,vr


pylab.savefig(sname)
#pylab.show()




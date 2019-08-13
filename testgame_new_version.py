#!/usr/bin/env python
from game_new_version import *
from scipy import *
import pylab, sys
import scipy.linalg as linalg

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Ellipse

random.seed(100)
def likemany(x):
    return like(x)
    #return map(like,x)

if sys.argv[1]=='gauss':
    def like(x):
        return -((x[0])**2+(x[1])**2/1.0- 0.0*x[0]*x[1])/2.0
    ga=GMS(likemany,[1.0,2.0], [0.4, 0.5])
    #ga.N1=1000
    #ga.tweight=1.50
    #ga.mineffsamp=5000
    #sname='gauss.pdf'
    ga.run()


elif sys.argv[1]=='ring':
    def like(x):
        r2=x[0]**2+2*x[1]**2
        return -(r2-4.0)**2/(2*0.5**2)
    ga=GMS(likemany,[0.5,0.0],[0.2,0.3])
    ga.blow=2.0
    #ga.tweight=1.50
    #sname='ring.pdf'
    ga.run()

elif sys.argv[1]=='box':
    #Has to return a matrix instead of a single vector.
    def like(x):
        if (np.abs(x[0]) >1.0) or (np.abs(x[1]) >1.0):
            return -30
        else:
            return 0
    ga=GMS(likemany,[2.0,2.0],[0.3,0.3])
    #ga.tweight=1.5
    #ga.N1=1000
    ga.run()
    #sname='box.pdf'
else:
    sys.exit("define")


def add_plot(x, pylab, bar=False, vmin=None, vmax=None):
    pylab.imshow(x, interpolation='nearest', origin='lower left',
                extent=[cmin,cmax,cmin,cmax], cmap='jet', vmin=vmin, vmax=vmax)
    if bar: pylab.colorbar()



plot_conver = True
if plot_conver:

    fig = plt.figure(figsize=(10, 9))


    lG = len(ga.GaussList)
    ngaus = sp.arange(lG)
    lmn = int(len(ga.allmean) / 2.)
 #   cmap = plt.cm.rainbow
 #   norm = matplotlib.colors.Normalize(vmin=lmn, vmax=lG)
 #   colors = 2 * sp.arange(lG) + sp.rand(lG)


    lmn = 0
    fig.add_subplot(221)
    plt.scatter(ngaus[lmn:], zip(*ga.allmean[lmn:])[0],  s=50, c='blue', alpha=0.75, marker='p')
    plt.scatter(ngaus[lmn:], zip(*ga.allmean[lmn:])[1],  s=50, c='red', alpha=0.75)

    fig.add_subplot(222)
    plt.scatter(ngaus[lmn:], zip(*ga.allvar[lmn:])[0], s=50, c='blue', alpha=0.75, marker='p')
    plt.scatter(ngaus[lmn:], zip(*ga.allvar[lmn:])[1], s=50, c='red', alpha=0.75)

    fig.add_subplot(223)
    plt.scatter(ngaus[lmn:], ga.Neffsample[lmn:], s=50, c='red', alpha=0.75)

    fig.add_subplot(223)
    plt.scatter(ngaus[lmn:], ga.Neffsample[lmn:], s=50, c='red', alpha=0.75)

    fig.add_subplot(224)
    plt.scatter(ngaus, ga.allKL, s=50, c='red', alpha=0.75)

#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm.set_array([])
#    plt.colorbar(sm)

    plt.savefig('means.pdf')
    #plt.show()



plotting = True
if plotting:
    ## now we  plot
    Np = 100
    cmin, cmax = -5., 5.
    cstep = (cmax - cmin) / (1.0 * Np)

    #sys.exit("Seems no errors so far")
    xx = array([sa.positions[0] for sa in ga.SamplingList])
    yy = array([sa.positions[1] for sa in ga.SamplingList])
    ww = array([sa.weight for sa in ga.SamplingList])

    sums   = zeros((Np, Np))
    wsums  = zeros((Np, Np))

    for x, y, w in zip(xx, yy, ww):
        if (x<cmin) or (x>cmax) or (y<cmin) or (y>cmax):
            continue
        ix = int((x-cmin)/cstep)
        iy = int((y-cmin)/cstep)
        sums[iy, ix]  += 1.0
        wsums[iy, ix] += w

    x = y = np.arange(cmin, cmax, cstep)
    X, Y = np.meshgrid(x, y)
    trvals = np.exp(like([X, Y]))


    trvalsa = trvals/trvals.sum()
    wsumsa  = wsums/wsums.sum()
    diffp   = wsumsa - trvalsa
    vmax    = trvalsa.max()*1.1



    fig = plt.figure(figsize=(10, 9))

    ax = fig.add_subplot(221, aspect='equal')
    add_plot(sums, pylab)
    for G in ga.GaussList:
        ga.plot_ellipse(G, ax, 0, elipse=False)
    plt.xlim(cmin,cmax)
    plt.ylim(cmin,cmax)

    pylab.subplot(2,2,2)
    add_plot(wsumsa, pylab, vmin=0, vmax=vmax)

    pylab.subplot(2,2,3)
    add_plot(trvalsa, pylab, vmin=0, vmax=vmax)

    pylab.subplot(2,2,4)
    add_plot(diffp, pylab, bar=True)



    mx=(xx*ww).sum()/(ww.sum())
    vx=sqrt((xx**2*ww).sum()/(ww.sum())-mx*mx)
    my=(yy*ww).sum()/(ww.sum())
    vy=sqrt((yy**2*ww).sum()/(ww.sum())-my*my)
    rr=xx**2+yy**2
    mr=(rr*ww).sum()/(ww.sum())
    vr=sqrt((rr**2*ww).sum()/(ww.sum())-mr*mr)

    print 'xmean,xvar=',mx,vx
    print 'ymean,yvar=',my,vy
    print 'rmean,rvar=',mr,vr


    #pylab.savefig(sname)
    pylab.show()




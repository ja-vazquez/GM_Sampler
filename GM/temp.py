# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from mpi4py import MPI

def funct(x):
    return x**2

N= 1000000
x = np.arange(N)
#print x, map(funct, x)

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()


chunks = [[] for _ in np.arange(nprocs)]
n= np.ceil(float(len(x))/nprocs)

for i, chunk in enumerate(x):
    chunks[int(i//n)].append(chunk)
      
#print [map(funct, chuck) for chuck in chunks ]

scatter= comm.scatter(chunks, root=0)
result_per_node = map(funct, scatter)
result_gather   = comm.gather(result_per_node, root=0)

#final_result= np.array(result_gather)
#print final_result.flatten()  # only work if sublist have same # elements

if myrank == 0:
    final_result = [item for sublist in result_gather for item in sublist]
else:
    final_result = None

#print final_result 
#print comm.rank, scatter, result_per_node
#print result_gather
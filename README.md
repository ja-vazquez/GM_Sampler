# GM_Sampler
Massively parallelizable sampling algorithm
[Still under construction; main contributions by A. Slosar].


Here we present a new sampling method to be used in cosmology. 
Its main advantage over the standard MCMC 
sampling is that it is trivially paralellizable over larger number of 
nodes and with an appropriately chosen way of selecting covariance matrix 
might outperform the MCMC algorithm. 
It performs excellently in 2-dimensional toy examples and reproduces the 
results of standard algorithm in a realistic cosmological setting.

[Here is a more detailed and technical description.](https://github.com/slosar/GMSampler)

###Code

The files `game_simpleMC.py`, `driver_game.py`, `wqdriver_game.py`
should be incorporated into the [SimpleMC](https://github.com/ja-vazquez/SimpleMC) code to run it using realistic cosmological models (LCDM) and proper likelihoods (BAO, SNe, CMB).

###Toy model

There is also a toy model version, which can be run by


	python testgame.py like

*like* represents the likelihood to explore ={gauss, ring, box}, 
or you could build up your own likelihood function, as:

	def like(x):
		...
		...

Some toy models.
![](https://github.com/ja-vazquez/GM_Sampler/blob/master/gauss.jpg?raw=true =100x)
![](https://github.com/ja-vazquez/GM_Sampler/blob/master/ring.jpg?raw=true =100x)

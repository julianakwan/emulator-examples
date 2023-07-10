# Example codes for emulation
A repository for example codes to do Gaussian Process regression - these are all under development! 


emceegp.py uses the procedure detailed in <a href=https://adsabs.net/abs/2009ApJ...705..156H> Heitmann et al. (2009)</a> to build emulators. It will determine the PCA basis and then use MCMC to fit for the hyperparameters using <a href = https://emcee.readthedocs.io/en/stable/index.html > emcee </a>. Because of this, you will need to have emcee installed as well. It does not make predictions - you will need to take the predicted weights and PC basis functions plug them into a separate code. 

multires_gp.py uses heteroschedastic GP regression (using <a href = https://sheffieldml.github.io/GPy/> GPy </a>) to weight power spectra from different resolution simulations and combine them under one single GP. 


Please get in touch: julianakwan123@gmail.com if you have questions!

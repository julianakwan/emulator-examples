# Example codes for emulation
A repository for example codes to do Gaussian Process regression - these are all under development! 


emceegp.py uses the procedure detailed in Heitmann et al. (2009) to build emulators. It will determine the PCA basis and then use MCMC to fit for the hyperparameters using emcee. It does not make predictions - you will need to take the predicted weights and PC basis functions plug them into a separate code. 

multires_gp.py uses heteroschedastic GP regression (using GPy) to weight power spectra from different resolution simulations and combine them under one single GP. 

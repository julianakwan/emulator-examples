# emulator-examples
A repository for example codes to do Gaussian Process regression


emceegp.py uses the procedure detailed in Heitmann et al. (2009) to build emulators. It will determine the PCA basis and then use MCMC to fit for the hyperparameters using emcee. It does not make predictions - you will need to take the predicted weights and PC basis functions plug them into a separate code. 

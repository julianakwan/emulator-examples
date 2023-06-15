import numpy as np
import emcee
import CalcDistances

def svd_invert(A):
    U, S, V = np.linalg.svd(A,full_matrices=True)
    numdiag = len(S)
    keep = np.where(S>1e-6)
    newS = S[keep]*np.eye(len(S[keep]),len(S[keep]))
    newU = U[:,0:keep[0][-1]]
    newV = V[:,0:keep[0][-1]]
    Ainv = np.matmul(newV, np.matmul(np.linalg.inv(newS),np.transpose(newU)))

    return Ainv

def CalcPrior(rho_w, lambda_w, lambda_p):
    if (rho_w > 0).all() and (rho_w <= 1).all() and (lambda_w > 0).all() and (lambda_w < 10).all() and 1. < lambda_p < 1e10:
        return 0.0
    
    return -np.inf

def CalcCovMat(rho_w, lambda_w, lambda_p, nparams, numPC, nmodels, distmat):

    CovMatInd = np.ones((nmodels,nmodels,numPC))

# These are the old assignments; you can check GetNames.py as to how the matrices are ordered.
#    rho = AssignRho(nparams, numPC, block) 
#    lambda_w = AssignLambda(numPC, block)

    for h in range(0,numPC):
        for k in range(0,nparams):
            CovMatInd[:,:,h] = CovMatInd[:,:,h]*rho_w[k,h]**distmat[:,:,k]

    CovMat = np.zeros((numPC*nmodels,numPC*nmodels))

    for i in range (0,numPC):
        CovMat[i*nmodels:(i+1)*nmodels,i*nmodels:(i+1)*nmodels] = CovMatInd[:,:,i]/lambda_w[i]

#Use kron((eye(numPC),CovMatInd[:,:,i]))
        

    CovMat = CovMat + np.eye(nmodels*numPC)*1e-8

    return CovMat


def CalcLike(hparams, nparams, numPC, nmodels, nobs, distmat, what, invPtrP, Phi, pstar):
    # hparams contains rho_w, lambda_w and lambda_p in that order

    rho_w = np.reshape(hparams[0:numPC*nparams],(nparams,numPC))
    lambda_w = hparams[numPC*nparams:numPC*nparams+numPC]
    lambda_p = np.exp(hparams[numPC*nparams+numPC])


    #I think these should be parameters also? But in a different likelihood calculation
    #Maybe not necessary to do a full MCMC but add them on to the likelihood
    #to get the final posterior.
    
    
    a_rho_w = 1.
    b_rho_w = 0.1
    a_lambda_w = 5.
    b_lambda_w = 5.
    a_lambda_p = 1.
    b_lambda_p = 1e-4
    a_lambda_p_dash = a_lambda_p + 0.5*nmodels*(nobs-numPC)
    b_lambda_p_dash = b_lambda_p + 0.5*np.dot(np.transpose(pstar), (pstar-np.dot(Phi,what)))




    eps = 1e-8

    #These are the posteriors but what are their likelihoods?
    #Also they should be in separate likelihood definition
    

    prior = CalcPrior(rho_w, lambda_w, lambda_p)

    if np.isfinite(prior):
        #Calculate covariance matrix
        CovMat = CalcCovMat(rho_w, lambda_w, lambda_p, nparams, numPC, nmodels, distmat)
        
        lambda_p_parts = np.log(lambda_p)*(a_lambda_p_dash-1)-b_lambda_p_dash*lambda_p
        lambda_w_parts = (a_lambda_w-1)*np.sum(np.log(lambda_w))-b_lambda_w*np.sum(lambda_w)
        rho_w_parts = np.sum(np.ones(np.shape(rho_w))-rho_w)
        rho_w_parts = (b_rho_w-1)*np.log(rho_w_parts)

        emu_like = lambda_p_parts + lambda_w_parts + rho_w_parts #these are the priors
#        print emu_like
        emu_like = emu_like - 0.5*np.dot(np.transpose(what),np.dot(np.linalg.pinv(CovMat+invPtrP/lambda_p),what)) - 0.5*np.log(np.linalg.det(CovMat+invPtrP/lambda_p)+eps)    
        return prior+emu_like
    else:
        return -np.inf    
#    print np.argwhere(np.isnan(CovMat))
    





if __name__ == "__main__":
    #Number of principal components 
    numPC = 5

    #Read in training data
    ysim = np.loadtxt("b1_smooth_sod_test.453.txt")
#    ysim = np.loadtxt("hmf.sod.499.txt")
#    ysim = np.log10(ysim)
    #Ordering assumes one model per column; this is important when we are doing the PC decomposition
    ysim = np.transpose(ysim)
    
    
    #Number of output points
    nobs = np.shape(ysim)[1]
    
    #Read in design
    design = np.loadtxt("design_tier1.dat")
    nmodels = len(design)
    ntheta = np.shape(design)[1]

    #Normalize the design to be between 0,1
    design_max = np.max(design,0)
    design_min = np.min(design,0)
    design = (design-design_min)/(design_max-design_min)

    #Standardize the training data
    ymean = np.mean(ysim,1)
    ystd = ysim - np.transpose(np.tile(ymean,(nmodels,1)))
    ysimsd = np.sqrt(np.var(ystd,1))
    ystd = ystd/np.transpose(np.tile(ysimsd,(nmodels,1)))

    [U,S,V] = np.linalg.svd(ystd,full_matrices=True) #V is transposed w.r.t Matlab definition

    S = S*np.eye(len(S),len(S))
    
    phi = np.matmul(U[:,0:numPC],S[0:numPC,0:numPC])/np.sqrt(nmodels)

    #Be careful about the ordering; F is for Fortran style
    pstar = np.ravel(ystd,'F') 

    Phi = np.zeros([nobs*nmodels,nmodels*numPC])
    
    #for some reason the ordering is the transpose of the matlab case
    Phi = np.transpose(np.kron(np.eye(nmodels),phi[:,0])) 

    for i in range(1,numPC):
        Phi = np.concatenate((Phi, np.transpose(np.kron(np.eye(nmodels),phi[:,i]))),axis=1)

    PtrP = np.matmul(np.transpose(Phi),Phi)
    invPtrP = np.linalg.pinv(PtrP, rcond=1e-6)
    what = np.matmul(np.transpose(Phi),pstar)
    what = np.matmul(invPtrP,what)

    #"distance matrix" of parameter values
    DistMat = CalcDistances.CalcDistMat(design)

    #Calculate PC weights. NB: V is the transpose of usual Matlab definition
    weights = np.reshape(np.array(np.sqrt(nmodels)*np.transpose(V[:,0:numPC])).swapaxes(0,1) ,[1,numPC*nmodels])


    a_rho_w = 1.
    b_rho_w = 0.1
    a_lambda_w = 5.
    b_lambda_w = 5.
    a_lambda_p = 1.
    b_lambda_p = 1e-4
    a_lambda_p_dash = a_lambda_p + 0.5*nmodels*(nobs-numPC)
    b_lambda_p_dash = b_lambda_p + 0.5*np.dot(np.transpose(pstar), (pstar-np.dot(Phi,what)))

    np.savetxt('phi.b1.nPC5.453.dat',phi)
    np.savetxt('what.b1.nPC5.453.dat',what)


    ndim = numPC*ntheta+numPC+1
    nwalkers = 2*ndim
    rho_w = np.ones(numPC*ntheta)
    lambda_w = np.ones(numPC)
    lambda_p = 5.
    hparams = np.ones((numPC+1)*ntheta)
    hparams = np.insert(hparams, (numPC+1)*ntheta, 1e5)
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    for i in range(0,nwalkers):
        p0[i][ndim-1] += lambda_p
    sampler = emcee.EnsembleSampler(nwalkers, ndim, CalcLike, args = (ntheta, numPC, nmodels, nobs, DistMat, what, invPtrP, Phi, pstar),threads=8)
#    pos,prob,state=sampler.run_mcmc(p0,50000)
#    samples = sampler.chain[:,1000:,:].reshape((-1,ndim))
#    np.savetxt('testchain_full.txt',samples)

    f = open("b1.nPC5.453.dat", "w")
    f.close()

    for result in sampler.sample(p0, iterations=5000, storechain=False):
        position = result[0] #result has 3 components: pos, prob, state
        f = open("b1.nPC5.453.dat","a")
        np.savetxt(f, position)
        f.close()
    

    


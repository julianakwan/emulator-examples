import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def func(x,a,b,c):
        return a*np.exp(-b*x)+c
        

def exp_extrapolate(k,k_max,pk):
        fac = 0.8 # lower bound on spline is fac*k_max 

        k_want = k[(k<k_max)&(k>k_max*fac)]
        pk_want = pk[(k<k_max)&(k>k_max*fac)]

        fit,cov = curve_fit(func,k_want, pk_want)
        pk_ext = func(k[k>k_max],*fit)
                        
        
        return pk_ext

def spline_extrapolate(k,k_max,pk):

        fac = 0.9 # lower bound on spline is fac*k_max 

        k_want = k[(k<k_max)&(k>k_max*fac)]
        pk_want = pk[(k<k_max)&(k>k_max*fac)]

        spl = splrep(x=k_want, y=pk_want)   #spline the power spectrum between fac*kmax < k < kmax

        pk_ext = splev(k[k>k_max], spl)     #evaluate spline
        
        return pk_ext

def polynomial_extrapolate(k,k_max,pk):
        fac = 0.85 # lower bound on spline is fac*k_max 

        k_want = k[(k<k_max)&(k>k_max*fac)]
        pk_want = pk[(k<k_max)&(k>k_max*fac)]

        fit = np.polyfit(k_want, pk_want,1) #last argument specifies order of polynomial
        polyfunc = np.poly1d(fit)
        pk_ext = polyfunc(k[k>k_max])

        return pk_ext

if __name__ == '__main__':
        do_holdout = 0
        do_test = 1

        N = 150
        nparams = 10

        N_test = 200

        k = np.logspace(-3.,1., 200)


        BoxSize = np.array([350., 700.,1400.])
        n_fft = 2520
        k_max = np.pi*n_fft/BoxSize/2.
        k_min = 2.*np.pi/BoxSize
        delta_k = np.pi/BoxSize
        
#        x_train_full = np.loadtxt('/home/arijkwan/cosmosis/output/design/slhs_nested_3x50_w0_m0p7_m1p3_wa_m0p7_p0p7_with_running.txt')
        x_train_full = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/slhs_nested_3x50_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running.txt")
#        x_train_full = x_train_full[:,:-3]


        x_train_full[:,1] = x_train_full[:,1]/(x_train_full[:,0]*x_train_full[:,2]**2)   #build in terms of omega_bh^2/omega_mh^2
#        x_train_full = np.delete(x_train_full,9,axis=1)  #if an extra column of sigma8 is repeated
#        x_train_full = x_train_full[:100]


        #Normalize the design to be between 0,1
        design_max = np.array([0.4, 0.17, 0.80, 1.0, 0.9, -0.6,  0.5,  0.005, 0.03,  8.5])
        design_min = np.array([0.2, 0.14, 0.60, 0.94, 0.7, -1.2,-1.6,  0.000, -0.03, 7.0])


#        design_max = np.max(x_train_full,0)
#        design_min = np.min(x_train_full,0)

        x_train_full = (x_train_full-design_min)/(design_max-design_min)
        x_train_full = x_train_full[:,:nparams]        

#        y_train_full = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_hmcode_slhs_nested_3x50_running_joint_w0wa_constraint.txt')
#        pk_lin_train = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_lin_slhs_nested_3x50_running_joint_w0wa_constraint.txt')

#        y_train_full = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_hmcode_slhs_nested_3x50_running_w0_m1p3_m0p7_wa_m0p7_p0p7.txt')
#        pk_lin_train = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_lin_slhs_nested_3x50_running_w0_m1p3_m0p7_wa_m0p7_p0p7.txt')

        y_train_full = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/pk_hmcode2020_feedback_slhs_nested_3x50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt")
        pk_lin_train = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/pk_lin_camb2022_slhs_nested_3x50_running_w0_m1p2_m0p6_wa_m1p6_p0p5.txt")               
        
        
        y_train_full = y_train_full/pk_lin_train

###### for k values less than k_min make the ratio go to 1 #######
#        y_train_full[:50, k<k_min[0]] = 1
#        y_train_full[50:100, k<k_min[1]] = 1
#        y_train_full[100:, k<k_min[2]] = 1


#        [plt.plot(k, np.log10(y_train_full[i])) for i in range(150)]

#        plt.show()


        y_train_full = np.log10(y_train_full)        

###### for k values greater than k_max extrapolate the ratio #######
#        for i in range(50,100):
#                y_train_full[i,k>k_max[1]] = polynomial_extrapolate(k,k_max[1],y_train_full[i])


#        for i in range(100,150):
#                y_train_full[i,k>k_max[2]] = polynomial_extrapolate(k,k_max[2],y_train_full[i])

#        [plt.plot(k, y_train_full[i]) for i in range(150)]                
#        plt.show()
                
        nmodels, nobs = np.shape(y_train_full)
        

        #Standardize the training data
        y_train_mean_full = np.mean(y_train_full,0)
        y_train_std_full = y_train_full - np.tile(y_train_mean_full,(nmodels,1))
        y_train_sd_full = np.sqrt(np.var(y_train_std_full,0))
        #In case of zeroes, set values to 1.
        y_train_sd_full[np.where(y_train_sd_full==0)]=1
        y_train_std_full = y_train_std_full/np.tile(y_train_sd_full,(nmodels,1))

        kern_low = GPy.kern.Matern52(input_dim=nparams,ARD=True)
        kern_high = GPy.kern.Matern52(input_dim=nparams,ARD=True)        
        kern = GPy.kern.Matern52(input_dim=nparams, ARD=True)


        
        #Regular homoscedastic GP model
#        GP_low = GPy.models.GPRegression(x_train_full, y_train_std_full, kernel=kern)
        GP = GPy.models.GPRegression(x_train_full, y_train_std_full, kernel=kern)
        GP.optimize(messages=True, max_iters=10000)
        
        #Heterscedastic GP model
#        y_train_std_high = y_train_std_full[:,k > np.min(k_max)]
#        y_train_std_low = y_train_std_full[:,k < np.min(k_max)]
#        GP_low = GPy.models.GPHeteroscedasticRegression(x_train_full, y_train_std_low, kernel=kern_low)
#        GP_high = GPy.models.GPRegression(x_train_full[:50], y_train_std_high[:50], kernel=kern_high)

#        GP_high = GPy.models.GPHeteroscedasticRegression(x_train_full, y_train_std_high, kernel=kern_high)        
#        GP_high.het_Gauss.variance[:50] = 0.01    #hi-res models
#        GP_high.het_Gauss.variance[50:100] = 1    #intermediate-res models
#        GP_high.het_Gauss.variance[100:] = 10    #low-res models
#        GP_high.het_Gauss.variance[:50].fix()  #do not fit errors for high res models

        

#        GP_low.optimize(messages=True, max_iters=10000) 
#        GP_high.optimize(messages=True, max_iters=10000)

      
        
        if (do_test == 1):

                fig, ax = plt.subplots()

#                x_test = np.loadtxt("/home/arijkwan/cosmosis/output/design/design_olhs_n100_w0waCDM.txt".format(N_test))   
#                x_test = np.loadtxt('/home/arijkwan/cosmosis/output/design/design_n{:d}_running_joint_w0wa_constraint.txt'.format(N_test))   #full param range test design
                x_test = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/test_design_n1000_w0_m0p6_m1p2_wa_m1p6_p0p5_with_running.txt")

                x_test[:,1] = x_test[:,1]/(x_test[:,0]*x_test[:,2]**2) #convert Omega_b h^2 to Omega_b/Omega_m

                x_test = (x_test-design_min)/(design_max-design_min)

                x_test = x_test[:,:nparams]

#                y_test = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_hmcode_n{:d}_running_joint_w0wa_constraint.txt'.format(N_test))    #full param range test models
#                pk_lin_test = np.loadtxt('/home/arijkwan/cosmosis/output/design/pk_lin_n{:d}_running_joint_w0wa_constraint.txt'.format(N_test))  #full param range test models (linear)

#                y_test = np.loadtxt("/home/arijkwan/cosmosis/output/design/pk_hmcode_olhs_n{:d}_w0waCDM.txt".format(N_test))
#                pk_lin_test = np.loadtxt("/home/arijkwan/cosmosis/output/design/pk_lin_olhs_n{:d}_w0waCDM.txt".format(N_test))

                y_test = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/pk_hmcode2020_feedback_n1000_test_running_w0_m1p3_m0p7_wa_m1p6_p0p5.txt")
                pk_lin_test = np.loadtxt("/home/arijkwan/GPemulate/bahamas/design/pk_lin_camb2022_n1000_test_running_w0_m1p3_m0p7_wa_m1p6_p0p5.txt")               
                
                y_test = y_test/pk_lin_test

                
#                y_test = y_test[(x_test[:,1]>0) &(x_test[:,1]<1)]
#                x_test = x_test[(x_test[:,1]>0) &(x_test[:,1]<1)]

                y_test = y_test[(x_test[:,6]>0) &(x_test[:,6]<1)]
                x_test = x_test[(x_test[:,6]>0) &(x_test[:,6]<1)]

                y_test = y_test[(x_test[:,5]>0) &(x_test[:,5]<1)]
                x_test = x_test[(x_test[:,5]>0) &(x_test[:,5]<1)]
                
                print(len(x_test))

                
                ratio = np.zeros([len(x_test),nobs])
                mse = np.zeros([len(x_test),nobs])
                for i in range(0,len(x_test)):
#                        weights_high, var = GP_high.predict(x_test[i].reshape(-1,1).T)
#                        weights_low, var = GP_low._raw_predict(x_test[i].reshape(-1,1).T)
#                        weights_high, var = GP_high._raw_predict(x_test[i].reshape(-1,1).T)

#                        y_pred_low = y_train_mean_full[k < np.min(k_max)] + y_train_sd_full[k < np.min(k_max)]*weights_low
#                        y_pred_high = y_train_mean_full[k > np.min(k_max)] + y_train_sd_full[k > np.min(k_max)]*weights_high
#                        y_pred = np.concatenate([y_pred_low, y_pred_high],axis=1)
                        
                        weights,var = GP.predict(x_test[i].reshape(-1,1).T)
                        y_pred = y_train_mean_full + y_train_sd_full*weights
                        y_pred = 10**y_pred[0,:]
                        ratio[i] = y_pred/y_test[i]
                        ax.plot(k, ratio[i])



                ax.plot(k, np.ones([nobs]), color='black')
                ax.plot(k, np.ones([nobs])*0.99, color='black', linestyle='--')
                ax.plot(k, np.ones([nobs])*1.01, color='black', linestyle='--')
                ax.plot(k, np.median(ratio,0), color='black', linewidth=2, label='median')
                ax.plot(k, np.quantile(ratio, 0.32, axis=0), color='black', linestyle='--')
                ax.plot(k, np.quantile(ratio, 0.68, axis=0), label = r'68% interval', color='black', linestyle='--')
                
#                ax.axvline(x=k_max[1], label='k={:4.2f}'.format(k_max[1]), color='r', linestyle='--')
#                ax.axvline(x=k_max[2], label='k={:4.2f}'.format(k_max[2]), color='r', linestyle=':')

                ax.set_xlabel('k [h/Mpc]')
                ax.set_ylabel('Prediction/True')
                ax.set_xscale('log')
#                ax.legend()
                plt.ylim(0.9,1.1)
#                plt.savefig('/home/arijkwan/GPemulate/bahamas/design/design_slhs_nested_3x50_running_joint_w0wa_constraint_w0waCDM_test.png'.format(N))
                plt.savefig('/home/arijkwan/GPemulate/bahamas/design/design_slhs_nested_3x50_running_w0_m1p3_m0p7_wa_m1p6_p0p5_dmo.png')
#                plt.savefig('/home/arijkwan/GPemulate/bahamas/design/design_slhs_nested_3x50_running_w0_m1p3_m0p7_wa_m1p6_p0p5_test_n1000_new_hmcode2020_feedback.png')                                
                plt.close()

        
        if (do_holdout==1):
                ratio = np.zeros([nmodels,nobs])  
                for holdout_no in range(0,nmodels):
                        x_train = np.delete(x_train_full,holdout_no,axis=0)                
                        y_train = np.delete(y_train_full, holdout_no, axis=0)
                
                        #Standardize the training data
                        y_train_mean = np.mean(y_train,0)
                        y_train_std = y_train - np.tile(y_train_mean,(nmodels-1,1))
                        y_train_sd = np.sqrt(np.var(y_train_std,0))
                        #In case of zeroes, set values to 1. 
                        y_train_sd[np.where(y_train_sd==0)]=1.
                        y_train_std = y_train_std/np.tile(y_train_sd,(nmodels-1,1))

        
                        kern = GPy.kern.RBF(input_dim=nparams,ARD=True)        
                        GP = GPy.models.GPRegression(x_train, y_train_std, kernel=kern)

                        GP.optimize(messages=True) # optimize sum of log-likelihood of experts 
                        weights, var = GP.predict(x_train_full[holdout_no].reshape(-1,1).T)

                        y_pred = y_train_mean + y_train_sd*weights

                        ratio[holdout_no] = 10**y_pred/10**y_train_full[holdout_no]
                        plt.plot(k, ratio[holdout_no])


                plt.ylim(0.9,1.1)
                plt.plot(k, np.ones([nobs]), color='black')
                plt.plot(k, np.ones([nobs])*0.99, color='black', linestyle='--')
                plt.plot(k, np.ones([nobs])*1.01, color='black', linestyle='--')

                plt.xscale('log')
                plt.xlabel('k [h/Mpc]')
                plt.ylabel('Prediction/True')
                plt.savefig('/home/arijkwan/GPemulate/bahamas/design/design_n{:d}_w0wa_constraint_holdout_test.png'.format(N))
                plt.show()
        


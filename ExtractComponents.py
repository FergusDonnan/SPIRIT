import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib.ticker as mticker
from matplotlib import gridspec
from matplotlib.colors import LogNorm, PowerNorm
import pandas as pd



Objs = ['eso137', 'eso420', 'ic5063', 'mcg05', 'ngc3081', 'ngc3227', 'ngc4051', 'ngc5506', 'ngc5728', 'ngc7172', 'ngc7582']

cutoff_T = 300 #300K
cutoff_tau = 1.0




######################

ls, S = np.loadtxt('./Ext.Curves/D23ExtCurve.txt', unpack=True)
ls_Em, Em = np.loadtxt('./Emissivity/D24Emissivity.txt', unpack=True)



def B_nu(x, A, T, alpha_sil, jax=False):
    
    c=299792458.0
    h=6.62607004e-34
    k = 1.38064852e-23

   # em = (1.0-alpha_sil)*jnp.interp(x, x_, em1) + alpha_sil*jnp.interp(x, x_, em2)
    em = jnp.interp(x, ls_Em, Em)
    x = x*1e-6 #micron to metres
    l_peak = 2.897771e-3/T
    nu = c/x
    nu_peak = c/l_peak


    if (jax==False):
        norm = ((nu_peak**3)/(np.exp(h*nu_peak/(k*T))-1.0))
        return (A* (nu**3)/(np.exp(h*nu/(k*T))-1.0))/norm   * em
    else:
        norm = ((nu_peak**3)/(jnp.exp(h*nu_peak/(k*T))-1.0)) 
        return (A* (nu**3)/(jnp.exp(h*nu/(k*T))-1.0))/norm * em 


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)
    

def round_to_1sf(x, n):
    if (np.isnan(x)==True):
        return 0.0
    
    try:
        return np.round(x, -int(np.floor(np.log10(x))) + (n - 1))
    except:
        return np.nan



fig = plt.figure(figsize=(5, int(4*len(Objs))))
gs = gridspec.GridSpec(len(Objs), 1, figure=fig, hspace = 0.6)

for k in range(len(Objs)):

	Obj = Objs[k]
	Dir_indx = 'Quick'
	x_d, y_d, e_d = np.loadtxt('./Data/'+Obj+'.txt', unpack=True, usecols=[0,1,2])


	y_d = y_d[x_d>1.5]
	e_d = e_d[x_d>1.5]
	x_d = x_d[x_d>1.5]
    
	SCALE = np.mean(y_d)

	x_d = x_d[y_d>1e-5]
	e_d = e_d[y_d>1e-5]
	y_d = y_d[y_d>1e-5]

	columns = ['Component', 'tau_9.8', 'tau_9.8_Err_up', 'tau_9.8_Err_low', 'T', 'T_Err_up', 'T_Err_low']
	df = pd.DataFrame(columns=columns)




	Psi_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-425:-25]
	scale = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-25]
	ext_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-20:]
	stl_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-24:-20]


	if (min(x_d)>4.5):
	    Psi_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-410:-10]
	    scale = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-10]
	    ext_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-6:]
	    stl_params = np.loadtxt('./Results/'+Obj+'/Differential/'+Dir_indx+'/params.txt', unpack=True, usecols=[1])[-9:-6]




	ax1 = plt.subplot(gs[k])
	#ax2 = plt.subplot(gs[int(2*k+1)])


	grid_size=20
	T = np.logspace(np.log10(35), np.log10(1500), grid_size)
	tau = np.logspace(np.log(0.05), np.log(15), grid_size, base=np.e)  
	norm1D = np.zeros(len(T))
	norm = np.empty((20, 20))
	norm2 = np.empty((20, 20))
	x = np.linspace(1.5, 28.0, 1000)
	Tau = np.interp(x, ls, S) 
	for i in range(len(T)):
	    for j in range(len(tau)):
	        B = B_nu(x, 1.0, T[i], 0.01)
	        ext = np.exp(-tau[j]*Tau)
	        norm[j, i] = np.trapz(B*ext, x)
	        norm2[j, i] = np.trapz(B, x)

	    norm1D[i] = np.trapz(B, x)

	norm_ = np.concatenate((-1.0*norm[::-1, :], norm), axis=0)
	norm_ratio = norm/norm2

	Psi = np.reshape(np.exp(Psi_params), (20,20))
	# Psi_dummy = np.full((20, 20), np.exp(-10.0))
	# tau_dummy = -1.0*tau[::-1]
	# Psi = np.concatenate((Psi_dummy, Psi), axis=0)

	# tau = np.concatenate((tau_dummy, tau))


	#fig,ax = plt.subplots(1, 2, figsize = (5,2))
	print(tau)
	pcol = ax1.pcolormesh(T, tau, Psi*norm_ratio,  antialiased=True, linewidth=0, rasterized=True, norm = PowerNorm(gamma=0.5))
	# print(tau[Psi==np.max(Psi)])
	pcol.set_edgecolor('face')

	#fig.suptitle(title)
	ax1.set_xlabel('$\log_{10}(T$ (K))')
	ax1.set_ylabel('$\ln(\\tau)$')



	ax1.set_xscale('log')
	ax1.set_yscale('log', base = np.e)
	ax1.set_xlabel('$T$ (K)')
	ax1.set_ylabel('$\\tau_{9.8}$ ')


	ax1.set_xticks([35, 100, 1000])
	ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
	ax1.set_yticks([ 0.1, 1.0, 10.0, 50.0])
	ax1.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
	ax1.set_xlim(35, 1500)
	ax1.set_ylim(0.05, 15)
	ax1.set_title(Obj)
	ax1.invert_yaxis()

	Psi_fit = Psi
	for j in range(np.shape(Psi_fit)[0]):
	    Psi_cut = np.copy(Psi_fit) 
	    Psi_cut[j+1:, :] = np.zeros(np.shape(Psi_cut[j+1:, :]))
	    r = np.sum(Psi_cut)/np.sum(Psi_fit)


    #€ax2.errorbar(x_d,y_d, ls='none', marker='.', ms=0.25)


	ax1.hlines(cutoff_tau, 35, 1500, ls='dashed', color='white')
	ax1.vlines(cutoff_T, cutoff_tau, 15, ls='dashed', color='white')


	ax1.annotate('A', (45, 12), color = 'white')
	ax1.annotate('B', (1200, 12), color = 'white')
	if (cutoff_tau>0.0):
		ax1.annotate('C', (45, 0.9*cutoff_tau), color = 'white')


	Psi_AGN = np.copy(Psi_fit)
	Psi_AGN[(tau<cutoff_tau), :] = np.zeros(np.shape(Psi_AGN[(tau<cutoff_tau), :]))

    #if (k==7):
	Psi_AGN[:, (T>cutoff_T)] = np.zeros(np.shape(Psi_AGN[:, (T>cutoff_T)]))
    #Psi_AGN_Err[:, (T>cutoff_T)] = np.zeros(np.shape(Psi_AGN_Err[:, (T>cutoff_T)]))

    # Psi_AGN[(tau<cutoff), :] = np.full(np.shape(Psi_AGN[(tau<cutoff), :]),  np.exp(-10.0))

	Psi_SF = np.copy(Psi_fit)
    #Psi_SF_Err = np.copy(Psi_Err)

    #Psi_SF[(tau>cutoff), :] = np.full(np.shape(Psi_SF[(tau>cutoff), :]),  np.exp(-10.0))
	Psi_SF[(tau>cutoff_tau), :] = np.zeros(np.shape(Psi_SF[(tau>cutoff_tau), :]))
    #Psi_SF_Err[(tau>cutoff_tau), :] = np.zeros(np.shape(Psi_SF_Err[(tau>cutoff_tau), :]))

	Psi_Polar = np.copy(Psi_fit)
    #Psi_Polar_Err = np.copy(Psi_Err)

    #Psi_SF[(tau>cutoff), :] = np.full(np.shape(Psi_SF[(tau>cutoff), :]),  np.exp(-10.0))
	Psi_Polar[(tau<cutoff_tau), :] = np.zeros(np.shape(Psi_Polar[(tau<cutoff_tau), :]))
	Psi_Polar[:, (T<cutoff_T)] = np.zeros(np.shape(Psi_Polar[:, (T<cutoff_T)]))
    #Psi_Polar_Err[(tau<cutoff_tau), :] = np.zeros(np.shape(Psi_Polar_Err[(tau<cutoff_tau), :]))
    #Psi_Polar_Err[:, (T<cutoff_T)] = np.zeros(np.shape(Psi_Polar_Err[:, (T<cutoff_T)]))




	w_x = np.sum(Psi_SF*norm_ratio, axis=0)
	w_y = np.sum(Psi_SF*norm_ratio, axis=1)
	mean_tau = weighted_percentile(tau, w_y, [0.16, 0.5, 0.84])# np.nanpercentile(mean_tau_s, [16, 50, 84])
	mean_T =weighted_percentile(T, w_x, [0.16, 0.5, 0.84])
	w_x = np.sum(Psi_AGN*norm_ratio, axis=0)
	w_y = np.sum(Psi_AGN*norm_ratio, axis=1)
	mean_tau2 = weighted_percentile(tau, w_y, [0.16, 0.5, 0.84])# np.nanpercentile(mean_tau_s, [16, 50, 84])
	mean_T2 =weighted_percentile(T, w_x, [0.16, 0.5, 0.84])

	w_x = np.sum(Psi_Polar*norm_ratio, axis=0)
	w_y = np.sum(Psi_Polar*norm_ratio, axis=1)
	mean_tau3 = weighted_percentile(tau, w_y, [0.16, 0.5, 0.84])# np.nanpercentile(mean_tau_s, [16, 50, 84])
	mean_T3 =weighted_percentile(T, w_x, [0.16, 0.5, 0.84])


	print(Obj)
	print('A: tau_9.8 = ', str(np.round(mean_tau2[1], 2)), '+', str(round_to_1sf(mean_tau2[2]-mean_tau2[1], 2)), '-', str(round_to_1sf(mean_tau2[1]-mean_tau2[0], 2)))
	print('A: T = ', str(np.round(mean_T2[1], 2)), '+', str(round_to_1sf(mean_T2[2]-mean_T2[1], 2)), '-', str(round_to_1sf(mean_T2[1]-mean_T2[0], 2)))
	print('')
	print('B: tau_9.8 = ', str(np.round(mean_tau3[1], 2)), '+', str(round_to_1sf(mean_tau3[2]-mean_tau3[1], 2)), '-', str(round_to_1sf(mean_tau3[1]-mean_tau3[0], 2)))
	print('B: T = ', str(np.round(mean_T3[1], 2)), '+', str(round_to_1sf(mean_T3[2]-mean_T3[1], 2)), '-', str(round_to_1sf(mean_T3[1]-mean_T3[0], 2)))
	if (cutoff_tau>0.0):

		print('')
		print('C: tau_9.8 = ', str(np.round(mean_tau[1], 2)), '+', str(round_to_1sf(mean_tau[2]-mean_tau[1], 2)), '-', str(round_to_1sf(mean_tau[1]-mean_tau[0], 2)))
		print('C: T = ', str(np.round(mean_T[1], 2)), '+', str(round_to_1sf(mean_T[2]-mean_T[1], 2)), '-', str(round_to_1sf(mean_T[1]-mean_T[0], 2)))



	df = pd.concat([df, pd.DataFrame([{'Component': 'A', 
	'tau_9.8':str(np.round(mean_tau2[1], 2)),
	 'tau_9.8_Err_up':str(round_to_1sf(mean_tau2[2]-mean_tau2[1], 2)), 
	 'tau_9.8_Err_low':str(round_to_1sf(mean_tau2[1]-mean_tau2[0], 2)), 
	 'T':str(np.round(mean_T2[1], 2)), 
	 'T_Err_up':str(round_to_1sf(mean_T2[2]-mean_T2[1], 2)), 
	 'T_Err_low':str(round_to_1sf(mean_T2[1]-mean_T2[0], 2))
                                      }])], ignore_index=True)
	df = pd.concat([df, pd.DataFrame([{'Component': 'B', 
	'tau_9.8':str(np.round(mean_tau3[1], 2)),
	 'tau_9.8_Err_up':str(round_to_1sf(mean_tau3[2]-mean_tau3[1], 2)), 
	 'tau_9.8_Err_low':str(round_to_1sf(mean_tau3[1]-mean_tau3[0], 2)), 
	 'T':str(np.round(mean_T3[1], 2)), 
	 'T_Err_up':str(round_to_1sf(mean_T3[2]-mean_T3[1], 2)), 
	 'T_Err_low':str(round_to_1sf(mean_T3[1]-mean_T3[0], 2))
                                      }])], ignore_index=True)

	if (cutoff_tau>0.0):

		df = pd.concat([df, pd.DataFrame([{'Component': 'C', 
	'tau_9.8':str(np.round(mean_tau[1], 2)),
	 'tau_9.8_Err_up':str(round_to_1sf(mean_tau[2]-mean_tau[1], 2)), 
	 'tau_9.8_Err_low':str(round_to_1sf(mean_tau[1]-mean_tau[0], 2)), 
	 'T':str(np.round(mean_T[1], 2)), 
	 'T_Err_up':str(round_to_1sf(mean_T[2]-mean_T[1], 2)), 
	 'T_Err_low':str(round_to_1sf(mean_T[1]-mean_T[0], 2))
                                      }])], ignore_index=True)

	df.to_csv(Obj+'_DustComponents.csv', index=False)



	# print('-------- SF -------- AGN -------- Polar --------')



	# print('$'+str(np.round(mean_tau[1], 2))+'^{+'+str(round_to_1sf(mean_tau[2]-mean_tau[1], 2))+'}_{-'+str(round_to_1sf(mean_tau[1]-mean_tau[0], 2))+'}$ & ' + 
    #      '$'+str(np.round(mean_T[1], 0))+'^{+'+str(round_to_1sf(mean_T[2]-mean_T[1], 2))+'}_{-'+str(round_to_1sf(mean_T[1]-mean_T[0], 2))+'}$ & ' + 
    #     '$'+str(np.round(mean_tau2[1], 2))+'^{+'+str(round_to_1sf(mean_tau2[2]-mean_tau2[1], 2))+'}_{-'+str(round_to_1sf(mean_tau2[1]-mean_tau2[0], 2))+'}$ & ' + 
    #      '$'+str(np.round(mean_T2[1], 0))+'^{+'+str(round_to_1sf(mean_T2[2]-mean_T2[1], 2))+'}_{-'+str(round_to_1sf(mean_T2[1]-mean_T2[0], 2))+'}$ & ' + 
    #     '$'+str(np.round(mean_tau3[1], 2))+'^{+'+str(round_to_1sf(mean_tau3[2]-mean_tau3[1], 2))+'}_{-'+str(round_to_1sf(mean_tau3[1]-mean_tau3[0], 2))+'}$ & ' + 
    #      '$'+str(np.round(mean_T3[1], 0))+'^{+'+str(round_to_1sf(mean_T3[2]-mean_T3[1], 2))+'}_{-'+str(round_to_1sf(mean_T3[1]-mean_T3[0], 2))+'}$ & ' 
    #      )

#plt.tight_layout()
fig.savefig('DustComponents.pdf')



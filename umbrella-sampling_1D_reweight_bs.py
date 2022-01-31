# Example illustrating the application of MBAR to compute a 1D PMF from an umbrella sampling simulation.
#
# The data represents an umbrella sampling simulation for the magnetization of the Ising model
# Adapted from one of the pymbar example scripts for 1D PMFs

import numpy as np # numerical array library
import pymbar # multistate Bennett acceptance ratio
import os
from pymbar import timeseries # timeseries analysis
from pymbar.utils import logsumexp
from glob import glob
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import brentq
import scipy.signal as signal
from scipy.signal import savgol_filter
kB = 1.0 # Boltzmann constant

# Parameters
temperature = 3.0 # assume a single temperature -- can be overridden with data from param file
N_max = 50000 # maximum number of snapshots/simulation
N_max_ref = 50000 # maximum number of snapshots/simulation
folders_top = glob("*/") # total number of temperatures
folders_1 = []
curdir = os.getcwd()
for i in range(len(folders_top)):
    os.chdir(curdir+'/'+folders_top[i])
    folders_bottom = glob("*/")
    for j in range(len(folders_bottom)):
        os.chdir(curdir+'/'+folders_top[i]+'/'+folders_bottom[j])
        folders_1.append(os.getcwd())
    os.chdir(curdir)
K = len(folders_1)

T_k = np.ones(K,float)*temperature # inital temperatures are all equal 
beta = 1.0 / (kB * temperature) # inverse temperature of simulations
mag_min = -1580 # min for magnetization
mag_max = 1580 # max for magnetization
mag_nbins = 395 # number of bins for magnetization

# Need to delete ext terms
# Allocate storage for simulation data
N_max = 50000
N_k = np.zeros([K], np.int32) # N_k[k] is the number of snapshots from umbrella simulation k
K_k = np.zeros([K], np.float64) # K_1_k[k] is the spring constant 1 for umbrella simulation k
mu_k = np.zeros([K], np.float64) # mu_k[k] is the chemical potential for umbrella simulation k
mag0_k = np.zeros([K], np.float64) # mag0_k[k] is the spring center location for umbrella simulation k
mag_kn = np.zeros([K,N_max], np.float64) # mag_kn[k,n] is the magnetization for snapshot n from umbrella simulation k
u_kn = np.zeros([K,N_max], np.float64) # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
g_k = np.zeros([K],np.float32);

# Read in umbrella spring constants and centers.
# Go through directories and read
umbrella_index = 0
for i in range(K):
    infile = open(folders_1[i]+'/param')
    for line in infile:
        line_strip = line.strip()
        if line_strip.startswith('harmon'):
            print(line_strip)
            line_split = line_strip.split()[1]
            K_k[i] = float(line_split) 
        if line_strip.startswith('window'):
            print(line_strip)
            line_split = line_strip.split()[1]
            mag0_k[i] = float(line_split) 
        if line_strip.startswith('T'):
            print(line_strip)
            line_split = line_strip.split()[1]
            T_k[i] = float(line_split) 
        if line_strip.startswith('h_external'):
            print(line_strip)
            line_split = line_strip.split()[1]
            mu_k[i] = float(line_split) 

beta_k = 1.0/(kB*T_k)   # beta factor for the different temperatures
print(beta_k)
print(mu_k)
if (np.min(T_k) == np.max(T_k)):
    DifferentTemperatures = False            # if all the temperatures are the same, then we don't have to read in energies.
    
# Read the simulation data
for i in range(K):
    k = i
    string_base = folders_1[i]
    # Read magnetization data.
    filename_mag = string_base+'/mbar_data.txt'
    print("Reading %s..." % filename_mag)
    infile = open(filename_mag, 'r')
    lines = infile.readlines()
    infile.close()
    # Parse data.
    n = 0
    for line in lines:
        tokens = line.split()
        mag = float(tokens[2]) # Magnetization
        u_kn[k,n] = float(tokens[1]) - float(tokens[0]) + mu_k[k]*mag # reduced potential energy without umbrella restraint and external field
        mag_kn[k,n] = mag            
        n += 1
    N_k[k] = n

    # Compute correlation times for potential energy and magnetization
    # timeseries.  If the temperatures differ, use energies to determine samples; otherwise, magnetization
            
    g_k[k] = timeseries.statisticalInefficiency(mag_kn[k,0:N_k[k]])
    print("Correlation time for set %5d is %10.3f" % (k,g_k[k]))
    indices = timeseries.subsampleCorrelatedData(mag_kn[k,0:N_k[k]], g=g_k[k]) 
    # Subsample data.
    N_k[k] = len(indices)
    u_kn[k,0:N_k[k]] = u_kn[k,indices]
    mag_kn[k,0:N_k[k]] = mag_kn[k,indices]

N_max = np.max(N_k) # shorten the array size

# At this point, start diverting from the usual path and allow a method that allows us to perform blocking/bootstrapping analysis
mag_n = mag_kn[0,0:N_k[0]] # mag_n[k] is the magnetization from some simulation snapshot
u_n = u_kn[0,0:N_k[0]] # u_n[k] is the potential energy from some snapshot that has mag value mag_n[k]
# Now append values
allN = N_k.sum()
for k in range(1,K):
    mag_n = np.append(mag_n, mag_kn[k,0:N_k[k]])
    u_n = np.append(u_n, u_kn[k,0:N_k[k]])

# Bootstrap time
N_bs = 20 # number of bootstrap samples
N_bs_start = 0 # index to start with outputs
np.random.seed(0)

# Some variable to skip output #
mbar_ref = []
mbar_count = 0

for N_ in range(N_bs_start,N_bs_start+N_bs):
    print("Iteration %d" % (N_))
    f_bs = open('mbar_'+str(N_)+'.txt', 'w')
    print("Iteration %d" % (N_), file=f_bs)
    # Select random samples
    g_reduction = 50    
     
    N_red = np.random.randint(allN, size=allN//g_reduction)
    N_red = np.sort(N_red)
    N_k_red = np.zeros([K], np.int32) 
    N_cumsum = np.cumsum(N_k)
    N_cumsum = np.hstack((np.array([0]), N_cumsum))
    # Determine N_k_red by binning
    for i in range(K):
        N_bin = (N_cumsum[i] <= N_red[:]) & (N_red[:] < N_cumsum[i+1])
        N_k_red[i] = N_bin.sum()

    u_n_red = u_n[N_red]
    mag_n_red = mag_n[N_red]
    u_kn_red = np.zeros((K, allN//g_reduction))

    for k in range(K):
        # Compute from umbrella center k
        dmag = mag_n_red[:] - mag0_k[k]

        # Compute energy of samples with respect to umbrella potential k
        u_kn_red[k,:] = beta_k[k]*(u_n_red[:] + (K_k[k]/2.0) * (dmag/1575.0)**2 -  mu_k[k]*mag_n_red[:])
        
    # Construct magnetization bins
    print("Binning data...", file=f_bs)
    delta_mag = (mag_max - mag_min) / float(mag_nbins)
    # compute bin centers
    bin_center_i_mag = np.zeros([mag_nbins], np.float64)
    for i in range(mag_nbins):
        bin_center_i_mag[i] = mag_min + delta_mag/2 + delta_mag * i
    # Bin data
    bin_n = np.zeros([allN//g_reduction], np.int64)+mag_nbins+10
    nbins = 0
    bin_counts = list()
    bin_centers = list() # bin_centers[i] is a tuple that gives the center of bin i
    for j in range(mag_nbins):
        # Determine which configurations lie in this bin
        in_bin = (bin_center_i_mag[j]-delta_mag/2 <= mag_n_red[:]) & (mag_n_red[:] < bin_center_i_mag[j]+delta_mag/2)
        # Count number of configurations in this bin
        bin_count = in_bin.sum()

        if (bin_count > 0):
            # store bin
            bin_centers.append(bin_center_i_mag[j])
            bin_counts.append( bin_count )
            # assign these conformations to the bin index
            bin_n[np.where(in_bin)[0]] = nbins
            # increment number of bins
            nbins += 1
           

    # Get total number of things that were binned
    bin_counts_np = np.array(bin_counts)
    bin_count_total = bin_counts_np.sum()
    bin_count_ideal = allN  

    # Make array with total combinations of bin_center_i_mag and bin_center_i_mag
    bin_center_possible = np.zeros((mag_nbins,1))
    bin_center_empty = np.zeros((mag_nbins,1))
    for i in range(mag_nbins):
        bin_center_possible[i] = bin_center_i_mag[i]

    # Determine empty bins
    for i in range(nbins):
        for k in range(mag_nbins):
            if((bin_centers[i] == bin_center_i_mag[k])):
                bin_center_empty[k] = 1
     
    print("%d bins were populated:" % nbins, file=f_bs)
    for i in range(nbins):
       print("bin %5d (%6.5f) %12d conformations" % (i, bin_centers[i], bin_counts[i]), file=f_bs)
    print("%d empty bins" % (mag_nbins-nbins), file=f_bs)
    for j in range(mag_nbins):
        if(bin_center_empty[j] == 0):
            print("bin (%6.5f)" % (bin_center_possible[j]), file=f_bs)
    print("%d / %d data used" % (bin_count_total, bin_count_ideal), file=f_bs)        

    # Initialize MBAR.
    print("Running MBAR...", file=f_bs)
    if(mbar_count == 0):
        mbar = pymbar.MBAR(u_kn_red, N_k_red, verbose = True, relative_tolerance=1e-10)
        mbar_ref = mbar.f_k
        mbar_count = mbar_count+1
    else:
        mbar = pymbar.MBAR(u_kn_red, N_k_red, verbose = True, relative_tolerance=1e-10, initial_f_k=mbar_ref)

    print('At reweighting step', file=f_bs)
    # Now have weights, time to have some fun reweighting
    u_n_red_original = u_n_red.copy()
    T_targets_low = np.linspace(2.0,3.0,26)
    T_targets_high = np.linspace(3.025, 3.7, 28)
    T_targets = np.hstack((T_targets_low, T_targets_high))
    low_comp_storage = np.zeros(T_targets.shape)
    high_comp_storage = np.zeros(T_targets.shape)
    mu_1_storage = np.zeros(T_targets.shape)
    mu_2_storage = np.zeros(T_targets.shape)
    mu_storage = np.zeros(T_targets.shape)

    # Compute PMF in unbiased potential (in units of kT) at kT = 1
    (f_i, df_i) = mbar.computePMF(u_n_red, bin_n, nbins)

    # Show free energy and uncertainty of each occupied bin relative to lowest free energy
    print("1D PMF", file=f_bs)
    print("", file=f_bs)
    print("%8s %6s %8s %10s %10s" % ('bin', 'mass', 'N', 'f', 'df'), file=f_bs)
    for i in range(nbins):
       print('%8d %10.8e %8d %10.10e %10.10e' % (i, bin_centers[i], bin_counts[i], f_i[i], df_i[i]), file=f_bs)

    # Write out PMF to file
    f_ = open('free_energy_'+str(mag_nbins)+'_original_'+str(N_)+'.txt', 'w')
    print("PMF (in units of kT)", file=f_)
    print("%8s %6s %8s %10s %10s" % ('bin', 'mass', 'N', 'f', 'df'), file=f_)
    for i in range(nbins):
       print('%8d %10.8g %8d %16.16e %16.16e' % (i, bin_centers[i], bin_counts[i], f_i[i], df_i[i]), file=f_)
    f_.close()

    for j in range(len(T_targets)):
        print("Reweighting at temperature "+str(T_targets[j]), file=f_bs) 
        # reweight to temperature of interest 
        u_n_red = u_n_red_original.copy()
        beta_reweight = 1.0/(kB*T_targets[j])   # beta factor for the different temperatures
        u_n_red = beta_reweight*u_n_red

        # Compute PMF in unbiased potential (in units of kT) at kT = 1
        (f_i_base, df_i_base) = mbar.computePMF(u_n_red, bin_n, nbins)

        mu_low = -1.0
        mu_high = 1.0

        # Now have mu_low and mu_high, use a bounded method to find mu which causes
        # f_i(comp_low) \approx f_i(comp_high)
        # let's use scipy's minimize_scalar solver for this
        # Have to define a function that we want to operate on
        def free_diff_comp(mu, f_i_base, bin_centers, beta_reweight):
            f_i = f_i_base - beta_reweight*mu*bin_centers
            mid_comp = int(3.0*nbins/4.0)
            f_i_low_comp = f_i[0:mid_comp].min()
            f_i_high_comp = f_i[mid_comp:nbins].min()
            return f_i_high_comp-f_i_low_comp

        print("", file=f_bs)
        print("Finding mu_eq_1", file=f_bs)
            
        # Find minimum
        mu_eq_1 = brentq(free_diff_comp, a=mu_low, b=mu_high, args=(f_i_base, np.array(bin_centers), beta_reweight))          
        mu_1_storage[j] = mu_eq_1
        
        print("mu_eq_1 %17.17e"%(mu_eq_1), file=f_bs)    
        print("", file=f_bs)
            
        # Now output results
        # Reweight to mu_eq
        f_i = f_i_base.copy()
        f_i = f_i - beta_reweight*mu_eq_1*np.array(bin_centers)
        f_i -= f_i.min()

        # Show free energy and uncertainty of each occupied bin relative to lowest free energy
        print("1D PMF with mu_eq_1", file=f_bs)
        print("", file=f_bs)
        print("%8s %6s %8s %10s" % ('bin', 'mass', 'N', 'f'), file=f_bs)
        for i in range(nbins):
           print('%8d %10.8g %8d %10.8e' % (i, bin_centers[i], bin_counts[i], f_i[i]), file=f_bs)

        f_ = open('mu_eq_1_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("%17.17e"%(mu_eq_1), file=f_)
        f_.close()

        # Write out PMF to file
        f_ = open('pmf_eq_1_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("PMF with mu_eq_1 (in units of kT)", file=f_)
        print("%8s %6s %8s %10s" % ('bin', 'mass', 'N', 'f'), file=f_)
        for i in range(nbins):
           print('%8d %10.8g %8d %16.16e' % (i, bin_centers[i], bin_counts[i], f_i[i]), file=f_)
        f_.close()

        # Write out probability to file
        p_i=np.exp(-f_i-logsumexp(-f_i))
        f_ = open('p_i_eq_1_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("PMF with mu_eq_1 (in units of kT)", file=f_)
        print("%8s %6s %8s %10s" % ('bin', 'mass', 'N', 'p'), file=f_)
        for i in range(nbins):
           print('%8d %10.8g %8d %16.16e' % (i, bin_centers[i], bin_counts[i], p_i[i]), file=f_)
        f_.close()
           
        # Now do it such that areas under peaks are the same
        def free_diff_comp_area(mu, f_i_base, nbins, bin_centers, beta_reweight):
            f_i = f_i_base - beta_reweight*mu*bin_centers
            p_i=np.exp(-f_i-logsumexp(-f_i))
            # Determine mid_comp
            # Filter f_i to determine where to divide peak
            f_i_filter = savgol_filter(f_i, window_length=41, polyorder=3)
            f_i_filter_2 = savgol_filter(f_i_filter, window_length=41, polyorder=3)
            rel_max = signal.argrelmax(f_i_filter_2, order=10)
            # print rel_max
            npeak = nbins//2
            if(len(rel_max[0]) == 0):
                npeak = nbins//2
            else:
                npeak = signal.argrelmax(f_i_filter_2, order=10)[0].max()
            # As bin size is equal for now, can just do naive sum as equivalent to
            # midpoint rule barring a constant factor
            low_area = np.trapz(p_i[0:npeak], x = bin_centers[0:npeak])
            high_area = np.trapz(p_i[npeak:nbins], x = bin_centers[npeak:nbins])
            return high_area-low_area

        print("", file=f_bs)
        print("Finding mu_eq_2", file=f_bs)
            
        # Find minimum
        mu_eq_2 = brentq(free_diff_comp_area, a=mu_eq_1-0.05, b=mu_high+0.05, args=(f_i_base, nbins, np.array(bin_centers), beta_reweight))          
        mu_2_storage[j] = mu_eq_2
        
        print("mu_eq_2 %17.17e"%(mu_eq_2), file=f_bs)    
        print("", file=f_bs)
            
        # Now output results
        # Reweight to mu_eq
        f_i = f_i_base.copy()
        f_i = f_i - beta_reweight*mu_eq_2*np.array(bin_centers)        
        f_i -= f_i.min()

        # Show free energy and uncertainty of each occupied bin relative to lowest free energy
        print("1D PMF with mu_eq_2", file=f_bs)
        print("", file=f_bs)
        print("%8s %6s %8s %10s %10s" % ('bin', 'mass', 'N', 'f', 'df'), file=f_bs)
        for i in range(nbins):
           print('%8d %10.8g %8d %10.8e %10.8e' % (i, bin_centers[i], bin_counts[i], f_i[i], df_i[i]), file=f_bs)

        f_ = open('mu_eq_2_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("%17.17e"%(mu_eq_2), file=f_)
        f_.close()

        # Write out PMF to file
        f_ = open('pmf_eq_2_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("PMF with mu_eq_2 (in units of kT)", file=f_)
        print("%8s %6s %8s %10s %10s" % ('bin', 'mass', 'N', 'f', 'df'), file=f_)
        for i in range(nbins):
           print('%8d %10.8g %8d %16.16e %16.16e' % (i, bin_centers[i], bin_counts[i], f_i[i], df_i[i]), file=f_)
        f_.close()
           
        # Get compositions
        p_i=np.exp(-f_i-logsumexp(-f_i))
        f_ = open('p_i_eq_2_'+str(mag_nbins)+'_'+str(T_targets[j])+'_'+str(N_)+'.txt', 'w')
        print("PMF with mu_eq_1 (in units of kT)", file=f_)
        print("%8s %6s %8s %10s" % ('bin', 'mass', 'N', 'p'), file=f_)
        for i in range(nbins):
           print('%8d %10.8g %8d %16.16e' % (i, bin_centers[i], bin_counts[i], p_i[i]), file=f_)
        f_.close()
        # Determine mid_comp
        f_i_filter = savgol_filter(f_i, window_length=41, polyorder=3)
        f_i_filter_2 = savgol_filter(f_i_filter, window_length=41, polyorder=3)
        rel_max = signal.argrelmax(f_i_filter_2, order=10)
        npeak = nbins//2
        if(len(rel_max[0]) == 0):
            npeak = nbins//2
            print('Weird divergence at %8d' % (j), file=f_bs)
        else:
            npeak = signal.argrelmax(f_i_filter_2, order=10)[0].max()   
        bin_centers_np = np.array(bin_centers)
        p_i_mass = bin_centers_np*p_i
        mass_avg = p_i_mass.sum()
        bin_closest = np.abs(bin_centers-mass_avg)    
        print("mass_avg %17.17e"%(mass_avg))
        # Now get entry that is closest to value
        mid_comp = np.argmin(bin_closest)
        mid_comp = npeak
        # Take
        low_comp = p_i_mass[0:mid_comp].sum()/p_i[0:mid_comp].sum()
        high_comp = p_i_mass[mid_comp:nbins].sum()/p_i[mid_comp:nbins].sum()
        print(low_comp, high_comp, T_targets[j])
        low_comp_storage[j] = low_comp/1575.0
        high_comp_storage[j] = high_comp/1575.0


    f_ = open('composition_reweight_'+str(N_)+'.txt', 'w')
    print('T phi_low phi_high', end=' ', file=f_)
    print("%10s %10s %10s" % ('T', 'phi_low', 'phi_high'), file=f_)
    for i in range(len(T_targets)):
        print('%16.16e %16.16e %16.16e' % (T_targets[i], low_comp_storage[i], high_comp_storage[i]), file=f_)
    f_.close()

    f_ = open('mu_reweight'+str(N_)+'.txt', 'w')
    print("%10s %10s %10s" % ('T', 'mu_peaks', 'mu_area'), file=f_)
    for i in range(len(T_targets)):
        print('%16.16e %16.16e %16.16e' % (T_targets[i], mu_1_storage[i], mu_2_storage[i]), file=f_)
    f_.close()
    f_bs.close()

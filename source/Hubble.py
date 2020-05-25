import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def fit_func_PL(P, alpha, beta):
    """
    Function that returns the result of the Cepheid PL relation.

    Parameters:
    P (float): Pulsation period of a star
    alpha (float): First free parameter of Cepheid PL relation
    beta (float): Second free parameter of Cepheid PL relation

    Returns:
    float: result of Cepheid PL relation for the given parameters

    """

    return alpha*np.log10(P) + beta


def fit_func_H(Dg, H):
    """
    Function that returns the Recession Velocity given Hubble's constant
    and the distance of a galaxy from the observer.

    Parameters:
    Dg (float): Distance of the galaxy
    H (float): Hubble Constant
    beta (float): Second free parameter of Cepheid PL relation

    Returns:
    float: Predicted Recession Velocity

    """

    return H*Dg


def get_approx_d_mod(file_name, alpha, beta, alpha_err, beta_err):
    """
    Returns the approximate distance modulus value of a given galaxy,
    estimating it from the collective individual stars whose data is available.
    The Inverse-Variance Weighted Mean method is used to do this.

    Parameters:
    file_name (str): File name for input data
    alpha (float): First free parameter of Cepheid PL relation
    beta (float): Second free parameter of Cepheid PL relation
    alpha_err (float): Uncertainty in alpha of Cepheid PL relation
    beta_err (float): Uncertainty in beta of Cepheid PL relation

    Returns:
    d_mod_mean (float): The approximated distance modulus value of the galaxy
    d_mod_mean_err (float): The uncertainty in this inverse-variance weighted mean

    """

    # LogP and apparent magnitude values from the stars are loaded
    logP, m = np.loadtxt(file_name, unpack=True, usecols=[1,2])

    # Absolute magnitude and uncertainty is calculated usingthe model
    M = alpha*logP + beta
    M_err = np.sqrt((logP**2*alpha_err**2) + (beta_err**2))

    # Distance modulus is calculate with the given data
    d_mod = m - M
    # Uncertainty in distance modulus is same as of that in Absolute Mag.

    # The inverse-variance weighted mean of all the stars is calculated
    d_mod_mean = np.sum(d_mod/M_err**2)/np.sum(1/M_err**2)

    # The corresponding uncertainty is calculate for the mean
    d_mod_mean_err = np.sqrt(1/np.sum(1/M_err**2))

    return d_mod_mean, d_mod_mean_err


def chi_squared(y, y_exp, y_err):
    """
    Returns the Chi2 statistic of a given set of predicted and observed
    values with a given uncertainty.

    Parameters:
    y (ndarray): Observed values of data
    y_exp (ndarray): Expected values of data
    y_err (ndarray): Uncertainty in each data point

    Returns:
    float: Chi2 statistics of the input data set

    """

    return np.sum(((y - y_exp)**2)/(y_err**2))


def plot_results(xs, ys, yerr, xlabel, ylabel):
    """
    Plots a graph with the given xs and ys along with error bars.

    Parameters:
    xs (ndarray): x values of plot
    ys (ndarray): y values of plot
    yerr (ndarray): Uncertainty in each data y data point
    xlabel (str): x-axis label
    ylabel (str): y-axis label

    """

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xs, ys, color='red', marker='o', markersize=5, linestyle='None')
    plt.errorbar(xs, ys, yerr, color='red', linewidth=2, linestyle='None')


def estimate_PL_rel():
    """
    Given the data for various sources of pulsating stars, the optimal parameters
    for which the Cepheid PL relation fits is returned with uncertainties.

    Returns:
    alpha (float): First free parameter of Cepheid PL relation
    beta (float): Second free parameter of Cepheid PL relation
    alpha_err (float): Uncertainty in alpha of Cepheid PL relation
    beta_err (float): Uncertainty in beta of Cepheid PL relation

    """

    print("-----------------------------------")
    print("DETERMINING THE CEPHEID PL RELATION")
    print("-----------------------------------\n")

    # Reading input data
    parallax, perr, period, m, A, Aerr = np.loadtxt('MW_Cepheids.dat', unpack=True, usecols=[1,2,3,4,5,6])
    
    # Calculating distance and uncert. in distance of source
    d_pc = 1000/parallax
    d_pc_err = np.sqrt((-1000/parallax**2)**2*perr**2)
    print("d_pc = ", d_pc)
    print("d_pc_err = ", d_pc_err)

    # Plotting results
    plot_results(parallax, d_pc, d_pc_err, 'Parallax', 'Distance Pc')
    plt.show()
    
    # Calculating value and uncert. of distance modulus
    d_mod = 5*np.log10(d_pc) - 5 + A
    d_mod_err = np.sqrt((5/(np.log(10)*d_pc))**2*d_pc_err**2 + Aerr**2)
    print("d_mod = ", d_mod)
    print("d_mod_err = ", d_mod_err)
    
    # Calculating Absolute Magnitude and uncert.
    abs_m = m - d_mod
    print("M = ", abs_m)
    # m_err assumed to be negligible
    abs_m_err = d_mod_err
    print("M err = ", abs_m_err)

    # Fitting with the given linear model
    params, covar = optimize.curve_fit(f=fit_func_PL, xdata=period, ydata=abs_m, 
                                                sigma=abs_m_err, absolute_sigma=True)

    # Optimal parameters
    alpha, beta = params

    # Uncert. in parameters
    alpha_err = np.sqrt(covar[0,0])
    beta_err = np.sqrt(covar[1,1])

    # Values of absolute magnitude predicted by model
    best_M = alpha*np.log10(period) + beta
    best_M_err = np.sqrt(np.log10(period)**2*alpha_err**2 + beta_err**2)
    print("Best M = ", best_M)
    print("Best M err = ", best_M_err)

    # Plotting results
    plot_results(period, best_M, best_M_err, 'Period P', 'Predicted abs. Magnitude M')
    plt.show()

    # Chi2 of the model
    best_chi2 = chi_squared(best_M, abs_m, abs_m_err)
    print()
    print("--------------------------------")
    print("Best alpha = ", alpha)
    print("Best beta = ", beta)
    print("Best alpha_err = ", alpha_err)
    print("Best beta_err = ", beta_err)
    print("Corrolation = ", covar[1,0])
    print("Best chi2 = ", best_chi2)

    # Reduced Chi2 of the model
    dof = len(period) - 2.0
    print("Best reduced chi2 = ", best_chi2/dof)
    print("--------------------------------")
    print()
    
    # Plotting calculated values of M with the predicted model (blue curve)
    plot_results(period, abs_m, abs_m_err, 'Period P', 'Absolute Magnitude M')

    # Plotting a smooth curve of the model
    period_smooth = np.arange(np.min(period), np.max(period), step=0.04)
    best_M_smooth = alpha*np.log10(period_smooth) + beta
    params_label = "alpha = "+str(alpha)+", beta = "+str(beta)
    plt.plot(period_smooth, best_M_smooth, color='blue', marker='None', linewidth=2, 
                                                        label=params_label, linestyle='-')
    plt.legend()
    plt.show()
    
    # Plotting alpha and beta with errors
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.plot(alpha, beta, color='red', marker='o', markersize=5, linestyle='None')
    plt.errorbar(alpha, beta, alpha_err, beta_err, color='red', linewidth=2, 
                                capsize=35, capthick=1, linestyle='None')
    plt.show()

    return alpha, beta, alpha_err, beta_err


def estimate_galaxy_distances(alpha, beta, alpha_err, beta_err):
    """
    Given the data for various sources of pulsating stars in various galazies, 
    the estimated distance of each galaxy is returned with its uncertainty.

    Parameters:
    alpha (float): First free parameter of Cepheid PL relation
    beta (float): Second free parameter of Cepheid PL relation
    alpha_err (float): Uncertainty in alpha of Cepheid PL relation
    beta_err (float): Uncertainty in beta of Cepheid PL relation
    

    Returns:
    dist (ndarray): Estimated distances of galaxies
    dist_err (ndarray): Uncertainties in estimated distances of galaxies

    """

    print("---------------------------")
    print("ESTIMATING GALAXY DISTANCES")
    print("---------------------------\n")

    # Loading galaxy data
    vr, A = np.loadtxt('galaxy_data.dat', unpack=True, usecols=[1,2])
    
    # Variables to calculate mu value
    approx_d_mod = []
    approx_d_mod_err = []
    
    # Calculating distance modulus for each galaxy
    for i in range(1,9):
        file_name = 'hst_gal'+str(i)+'_cepheids.dat'

        # Estimating distance modulus and uncertainty for given galaxy
        d_mod, d_mod_err = get_approx_d_mod(file_name, alpha, beta, alpha_err, beta_err)
        approx_d_mod.append(d_mod)
        approx_d_mod_err.append(d_mod_err)
    
    # Converting list of dist. mod and uncertainties to numpy arrays
    approx_d_mod = np.asarray(approx_d_mod)
    approx_d_mod_err = np.asarray(approx_d_mod_err)

    # Calculating estimated distance and uncertainty for each galaxy
    dist = 10**((approx_d_mod + 5 - A)/5)
    dist_err = np.sqrt(((np.log(10)/5)*dist)**2*approx_d_mod_err**2)

    # Converting to MPc
    dist = dist/(10**6)
    dist_err = dist_err/(10**6)
    print("dist = ", dist)
    print("dist_err = ", dist_err)
    print()
    
    # Plotting Distance of each galaxy with uncertainties
    xs = np.arange(1,9)
    plot_results(xs, dist, dist_err, 'Galaxy', 'Distance MPc')
    plt.show()

    return dist, dist_err


def estimate_hubble_const(dist, dist_err):
    """
    Given the distances and recession velocity of various galaxies,
    the Hubble Constant is estimated and returned with uncertainties.

    Parameters:
    dist (ndarray): Estimates of distances of galaxies
    dist_err (ndarray): Uncertainties in estimated distances

    Returns:
    Hubble_const (float): Estimated value of the Hubble Constant
    h_err (float): Uncertainty in estimation

    """

    print("--------------------------")
    print("ESTIMATING HUBBLE CONSTANT")
    print("--------------------------\n")

    # Recession velocity data loaded
    vr, A = np.loadtxt('galaxy_data.dat', unpack=True, usecols=[1,2])
    
    # Since no error is given, error of 100 is assumed then adjusted to fit reduced chi2 of 1
    vr_err = vr*0 + 100
    
    # Data is fit with the given Hubble's model
    hubble_const, pcov = optimize.curve_fit(f=fit_func_H, xdata=dist, ydata=vr,
                                                            sigma=vr_err, absolute_sigma=True)

    # Uncertainty in constant
    h_err = np.sqrt(pcov[0,0])
    print("Hubble const = ", hubble_const[0])
    print("Error in H = ", h_err)
    
    # Chi2 and reduced Chi2 is calculated
    dof = len(vr) - 1.0
    best_vr = hubble_const*dist
    best_chi2_h = chi_squared(best_vr, vr, vr_err)
    best_reduced_h = best_chi2_h/dof
    print("Best Chi^2 = ", best_chi2_h)
    print("Best Reduced Chi^2 = ", best_reduced_h)

    # Error in recession velocity is adjusted to match reduced Chi2 of 1
    vr_err = vr_err * np.sqrt(best_reduced_h)
    print("Adjusted VR err = ", vr_err[0])
    print()
    
    # Hubble's Const. and uncertainty is recalculated with adjusted vr_err
    hubble_const, pcov = optimize.curve_fit(f=fit_func_H, xdata=dist, ydata=vr,
                                                            sigma=vr_err, absolute_sigma=True)

    # Uncertainty in constant
    h_err = np.sqrt(pcov[0,0])
    print("Adjusted Hubble const = ", hubble_const[0])
    print("Adjusted Error in H = ", h_err)

    # New Chi2 with updated VR uncertainty
    best_chi2_h = chi_squared(best_vr, vr, vr_err)
    best_reduced_h = best_chi2_h/dof
    print("New Best Chi^2 = ", best_chi2_h)
    print("New Best Reduced Chi^2 = ", best_reduced_h)
    print()

    # Plotting Distance vs Velocity with estimated Hubble Constant
    plot_results(dist, dist*hubble_const, vr_err, 'Distance', 'Velocty')

    # Smoothing the predicted VR for plotting
    dist_smooth = np.arange(np.min(dist), np.max(dist), step=0.04)
    best_vr_smooth = hubble_const*dist_smooth
    plt.plot(dist_smooth, best_vr_smooth, color='blue', marker='None', linewidth=2,
                                    label="Hubble's const = "+str(hubble_const), linestyle='-')

    # Plotting 1-sigma interval ranges
    best_vr_smooth = (hubble_const+h_err)*dist_smooth
    plt.plot(dist_smooth, best_vr_smooth, color='black', marker='None', linewidth=2,
                                                    label='1-sigma interval', linestyle='dashed')
    best_vr_smooth = (hubble_const-h_err)*dist_smooth
    plt.plot(dist_smooth, best_vr_smooth, color='black', marker='None', linewidth=2,
                                                    label='1-sigma interval', linestyle='dashed')
    plt.legend()
    plt.show()

    return hubble_const, h_err, vr_err


def estimate_age(hubble_const, h_err):
    """
    Given the Hubble constant and the uncertainty in it,
    the age of the universe is estimated.

    Parameters:
    Hubble_const (float): Estimated value of the Hubble Constant
    h_err (float): Uncertainty in estimation

    """

    print("----------------------------------")
    print("ESTIMATING THE AGE OF THE UNIVERSE")
    print("----------------------------------\n")

    # Calculating the age and uncertainty
    #Converting the Hubble's Constant from km/s/MPc to years^-1
    hubble_const = 1.02271120232 * 10**-12 * hubble_const
    h_err = 1.02271120232 * 10**-12 * h_err

    # Calculating Age of Universe and uncertainty in gigayears
    tau = 10**-9/hubble_const
    err_tau = 10**-9*np.sqrt((-1/hubble_const**2)**2*(h_err**2))

    print("Age of universe in gigayears = ", tau)
    print("Error in gigayears = ", err_tau)
    print()


# Calibrating parameters of Cepheid PL relation with given data
alpha, beta, alpha_err, beta_err = estimate_PL_rel()

# Estimating the distance of various galaxies
dist, dist_err = estimate_galaxy_distances(alpha, beta, alpha_err, beta_err)

# Estimating the Hubble Constant
hubble_const, h_err, vr_err = estimate_hubble_const(dist, dist_err)

#Estimating the age of the universe
estimate_age(hubble_const, h_err)

"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

requirements:
numpy
tqdm
matplotlib
scipy
joblib
"""

# load relevant libraries
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import sys
###if sys.stderr.isatty():
###    from tqdm import tqdm
###else:
###    def tqdm(iterable, **kwargs):
###        return iterable
from tqdm import tqdm
###import warnings


def ivimN(bvalues, Dt, Fp, Dp, S0):
    # IVIM function in which we try to have equal variance in the different IVIM parameters; equal variance helps with certain fitting algorithms
    return S0 * ivimN_noS0(bvalues, Dt, Fp, Dp)


def ivimN_noS0(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def ivim(bvalues, Dt, Fp, Dp, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))


def order(Dt, Fp, Dp, S0=None):
    # function to reorder D* and D in case they were swapped during unconstraint fitting. Forces D* > D (Dp>Dt)
    if Dp < Dt:
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
    if S0 is None:
        return Dt, Fp, Dp
    else:
        return Dt, Fp, Dp, S0


def fit_segmented_array(bvalues, dw_data, mask_data = None, njobs=4, bounds=([0, 0, 0.005],[0.005, 0.7, 0.3]), cutoff=200):   ### mask_data added ### orig: bounds=([0, 0, 0.005],[0.005, 0.7, 0.2]), cutoff=75
    """
    This is an implementation of the segmented fit, in which we first estimate D using a curve fit to b-values>cutoff;
    then estimate f from the fitted S0 and the measured S0 and finally estimate D* while fixing D and f. This fit
    is done on an array.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: 2D Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :param cutoff: cutoff value for determining which data is taken along in fitting D
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # first we normalise the signal to S0
    S0 = np.mean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]
    
    # initialize empty arrays
    Dp = np.zeros(len(dw_data))
    Dt = np.zeros(len(dw_data))
    Fp = np.zeros(len(dw_data))
    for i in tqdm(range(len(dw_data)), position=0, leave=True):
        if (mask_data[i] != 0): ### line added
            # fill arrays with fit results on a per voxel base:
            Dt[i], Fp[i], Dp[i] = fit_segmented(bvalues, dw_data[i, :], bounds=bounds, cutoff=cutoff)
    return [Dt, Fp, Dp, S0]


def fit_segmented(bvalues, dw_data, bounds=([0, 0, 0.005],[0.005, 0.7, 0.3]), cutoff=200):    ### orig: bounds=([0, 0, 0.005],[0.005, 0.7, 0.2]), cutoff=75
    """
    This is an implementation of the segmented fit, in which we first estimate D using a curve fit to b-values>cutoff;
    then estimate f from the fitted S0 and the measured S0 and finally estimate D* while fixing D and f.
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :param cutoff: cutoff value for determining which data is taken along in fitting D
    :return Dt: Fitted D
    :return Fp: Fitted f
    :return Dp: Fitted Dp
    :return S0: Fitted S0
    """
    try:
        # determine high b-values and data for D
        high_b = bvalues[bvalues >= cutoff]
        high_dw_data = dw_data[bvalues >= cutoff]
        # correct the bounds. Note that S0 bounds determine the max and min of f
        bounds1 = ([bounds[0][0] * 1000., 1 - bounds[1][1]], [bounds[1][0] * 1000., 1. - bounds[0][1]])  
        # By bounding S0 like this, we effectively insert the boundaries of f
        # fit for S0' and D
        params, _ = curve_fit(lambda b, Dt, int: int * np.exp(-b * Dt / 1000), high_b, high_dw_data,
                              p0=(1, 1),
                              bounds=bounds1)
        Dt, Fp = params[0] / 1000, 1 - params[1]
        # remove the diffusion part to only keep the pseudo-diffusion
        dw_data_remaining = dw_data - (1 - Fp) * np.exp(-bvalues * Dt)
        bounds2 = (bounds[0][2], bounds[1][2])
        # fit for D*
        params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp), bvalues, dw_data_remaining, p0=(0.1), bounds=bounds2)
        Dp = params[0]
        return Dt, Fp, Dp
    except:
        # if fit fails, return zeros
        # print('segnetned fit failed')
        return 0., 0., 0.


def fit_least_squares_array(bvalues, dw_data, mask_data = None, S0_output=True, fitS0=True, njobs=4, ### mask_data added
                            bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])): ### orig bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]))
    """
    This is an implementation of the conventional IVIM fit. It is fitted in array form.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # normalise the data to S(value=0)
    S0 = np.nanmean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]

    # split up on whether we want S0 as output
    if S0_output:
        # Defining empty arrays
        Dp = np.zeros(len(dw_data))
        Dt = np.zeros(len(dw_data))
        Fp = np.zeros(len(dw_data))
        S0 = np.zeros(len(dw_data))
        # running in a single loop and filling arrays
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            if (mask_data[i] != 0): ### line added
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0,
                                                               bounds=bounds)
        return [Dt, Fp, Dp, S0]
    else:
        Dp = np.zeros(len(dw_data))
        Dt = np.zeros(len(dw_data))
        Fp = np.zeros(len(dw_data))
        for i in tqdm(range(len(dw_data))):
            Dt[i], Fp[i], Dp[i] = fit_least_squares(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0,
                                                    bounds=bounds)
        return [Dt, Fp, Dp]


def fit_least_squares(bvalues, dw_data, S0_output=False, fitS0=True,
                      bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])): ###orig bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3]))
    """
    This is an implementation of the conventional IVIM fit. It fits a single curve
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: Array with D in each voxel
    :return Fp: Array with f in each voxel
    :return Dp: Array with Dp in each voxel
    :return S0: Array with S0 in each voxel
    """
    try:
        if not fitS0:
            # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
            bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10],
                      [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10])
            params, _ = curve_fit(ivimN_noS0, bvalues, dw_data, p0=[1, 1, 0.1], bounds=bounds)
            S0 = 1
        else:
            # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
            bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
                      [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])
            params, _ = curve_fit(ivimN, bvalues, dw_data, p0=[1, 1, 0.1, 1], bounds=bounds)
            S0 = params[3]
        # correct for the rescaling of parameters
        Dt, Fp, Dp = params[0] / 1000, params[1] / 10, params[2] / 10
        # reorder output in case Dp<Dt
        if S0_output:
            return order(Dt, Fp, Dp, S0)
        else:
            return order(Dt, Fp, Dp)
    except:
        # if fit fails, then do a segmented fit instead
        # print('lsq fit failed, trying segmented')
        if S0_output:
            Dt, Fp, Dp = fit_segmented(bvalues, dw_data, bounds=bounds)
            return Dt, Fp, Dp, 1
        else:
            return fit_segmented(bvalues, dw_data)





def goodness_of_fit(bvalues, Dt, Fp, Dp, S0, dw_data):
    """
    Calculates the R-squared as a measure for goodness of fit.
    input parameters are
    :param b: 1D Array b-values
    :param Dt: 1D Array with fitted D
    :param Fp: 1D Array with fitted f
    :param Dp: 1D Array with fitted D*
    :param S0: 1D Array with fitted S0 (or ones)
    :param dw_data: 2D array containing data, as voxels x b-values
    :return R2: 1D Array with the R-squared for each voxel
    """
    # simulate the IVIM signal given the D, f, D* and S0
    try:
        datasim = ivim(np.tile(np.expand_dims(bvalues, axis=0), (len(Dt), 1)),
                       np.tile(np.expand_dims(Dt, axis=1), (1, len(bvalues))),
                       np.tile(np.expand_dims(Fp, axis=1), (1, len(bvalues))),
                       np.tile(np.expand_dims(Dp, axis=1), (1, len(bvalues))),
                       np.tile(np.expand_dims(S0, axis=1), (1, len(bvalues)))).astype('f')

        # calculate R-squared given the estimated IVIM signal and the data
        norm = np.mean(dw_data, axis=1)
        ss_tot = np.sum(np.square(dw_data - norm[:, None]), axis=1)
        ss_res = np.sum(np.square(dw_data - datasim), axis=1)
        R2 = 1 - (ss_res / ss_tot)  # R-squared
        adjusted_R2 = 1 - ((1 - R2) * (len(bvalues)) / (len(bvalues) - 4 - 1))
        R2[R2 < 0] = 0
        adjusted_R2[adjusted_R2 < 0] = 0
    except:
        datasim = ivim(bvalues, Dt, Fp, Dp, S0)
        norm = np.mean(dw_data)
        ss_tot = np.sum(np.square(dw_data - norm))
        ss_res = np.sum(np.square(dw_data - datasim))
        R2 = 1 - (ss_res / ss_tot)  # R-squared
        adjusted_R2 = 1 - ((1 - R2) * (len(bvalues)) / (len(bvalues) - 4 - 1))

        # from matplotlib import pyplot as plt
        # plt.figure(1)
        # vox=58885
        # plt.clf()
        # plt.plot(bvalues, datasim[vox], 'rx', markersize=5)
        # plt.plot(bvalues, dw_data[vox], 'bx', markersize=5)
        # plt.ion()
        # plt.show()
        # print(R2[vox])
    return R2, adjusted_R2


def MSE(bvalues, Dt, Fp, Dp, S0, dw_data):
    """
    Calculates the MSE as a measure for goodness of fit.
    input parameters are
    :param b: 1D Array b-values
    :param Dt: 1D Array with fitted D
    :param Fp: 1D Array with fitted f
    :param Dp: 1D Array with fitted D*
    :param S0: 1D Array with fitted S0 (or ones)
    :param dw_data: 2D array containing data, as voxels x b-values
    :return MSError: 1D Array with the R-squared for each voxel
    """
    # simulate the IVIM signal given the D, f, D* and S0
    datasim = ivim(np.tile(np.expand_dims(bvalues, axis=0), (len(Dt), 1)),
                   np.tile(np.expand_dims(Dt, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Fp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Dp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(S0, axis=1), (1, len(bvalues)))).astype('f')

    # calculate R-squared given the estimated IVIM signal and the data
    MSError = np.mean(np.square(dw_data - datasim), axis=1)  # R-squared
    return MSError
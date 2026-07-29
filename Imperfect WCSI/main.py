import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param

num_samples = 50

# num_samples = sys_param["num_CDF_samples"]



############ Find t_max from KL threshold #####################


sys_param["t_max"] = myf_find_t_max(sys_param)

print("KL threshold =", sys_param["KL_threshold"])

print("t_max =", sys_param["t_max"])


D01_nonrobust = np.zeros(num_samples)

D10_nonrobust = np.zeros(num_samples)

D01_robust = np.zeros(num_samples)

D10_robust = np.zeros(num_samples)


for ind1 in tqdm(range(0, num_samples)):

    param_channel = {}

    param_channel["seed_seq"] = ind1

    channel = myf_channel(
        sys_param,
        param_channel
    )
    #### True Willie channel ##########

    param_error = {}

    param_error["seed_seq"] = ind1

    h_W_true = myf_true_Willie_channel(
        sys_param,
        channel,
        param_error
    )

    ############ Non-robust design ############
    ########### Uses h_W_hat only ############
   

    solutions_nonrobust = myf_algorithm_nonrobust_imperfect_WCSI(
        sys_param,
        channel
    )


    D01_nonrobust[ind1], D10_nonrobust[ind1] = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_nonrobust["W_R_0"],
        solutions_nonrobust["W_R_1"]
    )


    # =====================================================
    # Robust design
    # Uses multiple perturbed Willie channels
    # =====================================================

    param_robust = {}

    param_robust["seed_seq"] = ind1

    solutions_robust = myf_algorithm_robust_imperfect_WCSI(
        sys_param,
        channel,
        param_robust
    )


    D01_robust[ind1], D10_robust[ind1] = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_robust["W_R_0"],
        solutions_robust["W_R_1"]
    )


    print(
        "sample =", ind1,
        "| D01 nonrobust =", D01_nonrobust[ind1],
        "| D01 robust =", D01_robust[ind1],
        "| D10 nonrobust =", D10_nonrobust[ind1],
        "| D10 robust =", D10_robust[ind1]
    )


# =====================================================
# Violation probability check
# =====================================================

KL_threshold = sys_param["KL_threshold"]

D01_nonrobust_valid = D01_nonrobust[~np.isnan(D01_nonrobust)]

D10_nonrobust_valid = D10_nonrobust[~np.isnan(D10_nonrobust)]

D01_robust_valid = D01_robust[~np.isnan(D01_robust)]

D10_robust_valid = D10_robust[~np.isnan(D10_robust)]


print("==========================================")

print("D(p0||p1) nonrobust violation ratio =",
      np.mean(D01_nonrobust_valid > KL_threshold))

print("D(p0||p1) robust violation ratio =",
      np.mean(D01_robust_valid > KL_threshold))

print("D(p1||p0) nonrobust violation ratio =",
      np.mean(D10_nonrobust_valid > KL_threshold))

print("D(p1||p0) robust violation ratio =",
      np.mean(D10_robust_valid > KL_threshold))

print("==========================================")


# =====================================================
# Plot Fig. 4
# =====================================================

myf_plot_Fig4_CDF(
    sys_param,
    D01_nonrobust,
    D10_nonrobust,
    D01_robust,
    D10_robust
)
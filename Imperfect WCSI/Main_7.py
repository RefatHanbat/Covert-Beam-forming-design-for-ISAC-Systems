import numpy as np

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 7 settings
# =====================================================

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


# Fixed Willie CSI error
sys_param["v_w"] = 0.005


# Fixed epsilon
sys_param["epsilon"] = 0.05

sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2


# Number of antennas candidate values
N_cand = np.arange(
    start=4,
    stop=9,
    step=1
)


# =====================================================
# Monte Carlo samples
# =====================================================

num_samples = 20

# Later:
# num_samples = 50
# num_samples = 100


# =====================================================
# KL thresholds for two cases
# =====================================================

t_max_D01 = myf_find_t_max_case(
    sys_param,
    "D01"
)

t_max_D10 = myf_find_t_max_case(
    sys_param,
    "D10"
)


print("===================================")

print("epsilon =", sys_param["epsilon"])

print("KL threshold =", sys_param["KL_threshold"])

print("v_w =", sys_param["v_w"])

print("t_max_D01 =", t_max_D01)

print("t_max_D10 =", t_max_D10)

print("===================================")


# =====================================================
# Result arrays
# =====================================================

MI_D01_result = np.zeros(np.size(N_cand))

MI_D10_result = np.zeros(np.size(N_cand))


for ind_N in range(0, np.size(N_cand)):


    # =====================================================
    # Update number of antennas
    # =====================================================

    sys_param["N"] = int(N_cand[ind_N])


    # =====================================================
    # Temporary arrays
    # =====================================================

    MI_D01_temp = np.zeros(num_samples)

    MI_D10_temp = np.zeros(num_samples)


    for ind_sample in tqdm(range(0, num_samples)):


        # =====================================================
        # Generate channel
        # Important:
        # Since N changes, channel must be generated after N update
        # =====================================================

        param_channel = {}

        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(
            sys_param,
            param_channel
        )


        # =====================================================
        # Robust design under D(p0 || p1)
        # =====================================================

        sys_param["t_max"] = t_max_D01

        param_robust = {}

        param_robust["seed_seq"] = ind_sample

        solutions_D01 = myf_algorithm_robust_imperfect_WCSI(
            sys_param,
            channel,
            param_robust
        )

        MI_D01_temp[ind_sample] = myf_radar_MI_from_I_R(
            sys_param,
            solutions_D01
        )


        # =====================================================
        # Robust design under D(p1 || p0)
        # =====================================================

        sys_param["t_max"] = t_max_D10

        param_robust = {}

        param_robust["seed_seq"] = ind_sample

        solutions_D10 = myf_algorithm_robust_imperfect_WCSI(
            sys_param,
            channel,
            param_robust
        )

        MI_D10_temp[ind_sample] = myf_radar_MI_from_I_R(
            sys_param,
            solutions_D10
        )


    # =====================================================
    # Average over samples
    # =====================================================

    MI_D01_result[ind_N] = np.nanmean(MI_D01_temp)

    MI_D10_result[ind_N] = np.nanmean(MI_D10_temp)


    print("===================================")

    print("N =", sys_param["N"])

    print("MI D01 =", MI_D01_result[ind_N])

    print("MI D10 =", MI_D10_result[ind_N])

    print("===================================")


# =====================================================
# Print final arrays
# =====================================================

print("===================================")

print("N_cand =", N_cand)

print("MI_D01_result =", MI_D01_result)

print("MI_D10_result =", MI_D10_result)

print("===================================")


# =====================================================
# Plot Fig. 7
# =====================================================

myf_plot_Fig7(
    sys_param,
    N_cand,
    MI_D01_result,
    MI_D10_result
)
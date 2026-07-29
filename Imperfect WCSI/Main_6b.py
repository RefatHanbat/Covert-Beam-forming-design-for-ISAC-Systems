import numpy as np

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 6(b) settings
# =====================================================

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


# Fig. 6(b): epsilon = 0.05
sys_param["epsilon"] = 0.05

sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2


# CSI error candidate values
v_w_cand = np.array([
    0.001,
    0.005,
    0.009,
    0.013,
    0.017
])


# Start small first
num_samples = 20

# Later, you can use:
# num_samples = 50
# num_samples = 100


# =====================================================
# Result arrays
# =====================================================

P_FA_D01_result = np.zeros(np.size(v_w_cand))

P_FA_D10_result = np.zeros(np.size(v_w_cand))

P_MD_D01_result = np.zeros(np.size(v_w_cand))

P_MD_D10_result = np.zeros(np.size(v_w_cand))


for ind_v in range(0, np.size(v_w_cand)):


    # =====================================================
    # Update CSI error value
    # =====================================================

    sys_param["v_w"] = v_w_cand[ind_v]


    # =====================================================
    # KL thresholds for two cases
    # =====================================================

    sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2


    # Case 1: D(p0 || p1) <= 2 epsilon^2
    t_max_D01 = myf_find_t_max_case(
        sys_param,
        "D01"
    )


    # Case 2: D(p1 || p0) <= 2 epsilon^2
    t_max_D10 = myf_find_t_max_case(
        sys_param,
        "D10"
    )


    # =====================================================
    # Temporary arrays
    # =====================================================

    P_FA_D01_temp = np.zeros(num_samples)

    P_FA_D10_temp = np.zeros(num_samples)

    P_MD_D01_temp = np.zeros(num_samples)

    P_MD_D10_temp = np.zeros(num_samples)


    for ind_sample in tqdm(range(0, num_samples)):


        # =====================================================
        # Generate estimated channel
        # =====================================================

        param_channel = {}

        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(
            sys_param,
            param_channel
        )


        # =====================================================
        # Generate true Willie channel
        # =====================================================

        param_error = {}

        param_error["seed_seq"] = ind_sample

        h_W_true = myf_true_Willie_channel(
            sys_param,
            channel,
            param_error
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

        P_FA_D01_temp[ind_sample], P_MD_D01_temp[ind_sample] = myf_detection_probability_Willie(
            sys_param,
            h_W_true,
            solutions_D01["W_R_0"],
            solutions_D01["W_R_1"]
        )


        # =====================================================
        # Robust design under D(p1 || p0)
        # =====================================================

        sys_param["t_max"] = t_max_D10

        param_robust = {}

        param_robust["seed_seq"] = ind_sample + 50000

        solutions_D10 = myf_algorithm_robust_imperfect_WCSI(
            sys_param,
            channel,
            param_robust
        )

        P_FA_D10_temp[ind_sample], P_MD_D10_temp[ind_sample] = myf_detection_probability_Willie(
            sys_param,
            h_W_true,
            solutions_D10["W_R_0"],
            solutions_D10["W_R_1"]
        )


    # =====================================================
    # Average over samples
    # =====================================================

    P_FA_D01_result[ind_v] = np.nanmean(P_FA_D01_temp)

    P_FA_D10_result[ind_v] = np.nanmean(P_FA_D10_temp)

    P_MD_D01_result[ind_v] = np.nanmean(P_MD_D01_temp)

    P_MD_D10_result[ind_v] = np.nanmean(P_MD_D10_temp)


    print("===================================")

    print("v_w =", sys_param["v_w"])

    print("epsilon =", sys_param["epsilon"])

    print("KL threshold =", sys_param["KL_threshold"])

    print("t_max_D01 =", t_max_D01)

    print("t_max_D10 =", t_max_D10)

    print("P_FA D01 =", P_FA_D01_result[ind_v])

    print("P_FA D10 =", P_FA_D10_result[ind_v])

    print("P_MD D01 =", P_MD_D01_result[ind_v])

    print("P_MD D10 =", P_MD_D10_result[ind_v])

    print("===================================")


# =====================================================
# Plot Fig. 6(b)
# =====================================================

myf_plot_Fig6b(
    sys_param,
    v_w_cand,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
)
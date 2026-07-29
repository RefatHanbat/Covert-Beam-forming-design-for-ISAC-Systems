import numpy as np

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 5 settings
# =====================================================

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["v_w"] = 0.001


epsilon_cand = np.arange(
    start=0.01,
    stop=0.10,
    step=0.02
)


num_samples = 20

# Later use:
# num_samples = 100
# num_samples = 1000


# =====================================================
# Result arrays
# =====================================================

MI_D01_result = np.zeros(np.size(epsilon_cand))

MI_D10_result = np.zeros(np.size(epsilon_cand))

P_FA_D01_result = np.zeros(np.size(epsilon_cand))

P_FA_D10_result = np.zeros(np.size(epsilon_cand))

P_MD_D01_result = np.zeros(np.size(epsilon_cand))

P_MD_D10_result = np.zeros(np.size(epsilon_cand))


for ind_epsilon in range(0, np.size(epsilon_cand)):

    epsilon = epsilon_cand[ind_epsilon]

    sys_param["epsilon"] = epsilon

    sys_param["KL_threshold"] = 2 * epsilon ** 2


    # =====================================================
    # Case 1: D(p0 || p1) <= 2 epsilon^2
    # =====================================================

    t_max_D01 = myf_find_t_max_case(
        sys_param,
        "D01"
    )


    # =====================================================
    # Case 2: D(p1 || p0) <= 2 epsilon^2
    # =====================================================

    t_max_D10 = myf_find_t_max_case(
        sys_param,
        "D10"
    )


    MI_D01_temp = np.zeros(num_samples)

    MI_D10_temp = np.zeros(num_samples)

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
        # Design under D(p0 || p1) constraint
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

        P_FA_D01_temp[ind_sample], P_MD_D01_temp[ind_sample] = myf_detection_probability_Willie(
            sys_param,
            h_W_true,
            solutions_D01["W_R_0"],
            solutions_D01["W_R_1"]
        )


        # =====================================================
        # Design under D(p1 || p0) constraint
        # =====================================================

        sys_param["t_max"] = t_max_D10

        param_robust = {}

        param_robust["seed_seq"] = ind_sample + 50000

        solutions_D10 = myf_algorithm_robust_imperfect_WCSI(
            sys_param,
            channel,
            param_robust
        )

        MI_D10_temp[ind_sample] = myf_radar_MI_from_I_R(
            sys_param,
            solutions_D10
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

    MI_D01_result[ind_epsilon] = np.nanmean(MI_D01_temp)

    MI_D10_result[ind_epsilon] = np.nanmean(MI_D10_temp)

    P_FA_D01_result[ind_epsilon] = np.nanmean(P_FA_D01_temp)

    P_FA_D10_result[ind_epsilon] = np.nanmean(P_FA_D10_temp)

    P_MD_D01_result[ind_epsilon] = np.nanmean(P_MD_D01_temp)

    P_MD_D10_result[ind_epsilon] = np.nanmean(P_MD_D10_temp)


    print("===================================")

    print("epsilon =", epsilon)

    print("KL threshold =", sys_param["KL_threshold"])

    print("t_max_D01 =", t_max_D01)

    print("t_max_D10 =", t_max_D10)

    print("MI D01 =", MI_D01_result[ind_epsilon])

    print("MI D10 =", MI_D10_result[ind_epsilon])

    print("P_FA D01 =", P_FA_D01_result[ind_epsilon])

    print("P_FA D10 =", P_FA_D10_result[ind_epsilon])

    print("P_MD D01 =", P_MD_D01_result[ind_epsilon])

    print("P_MD D10 =", P_MD_D10_result[ind_epsilon])

    print("===================================")


# =====================================================
# Plot Fig. 5
# =====================================================

myf_plot_Fig5(
    sys_param,
    epsilon_cand,
    MI_D01_result,
    MI_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
)
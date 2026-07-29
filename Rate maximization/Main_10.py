import numpy as np

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 10 settings
# =====================================================

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


# Fixed Willie CSI error
sys_param["v_w"] = 0.001


# Radar MI threshold for rate maximization
sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1


# Epsilon values in Fig. 10
epsilon_cand = np.array([
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
])


# =====================================================
# Monte Carlo samples
# =====================================================

num_samples = 1000

# Later:
# num_samples = 50
# num_samples = 100


# =====================================================
# Result arrays
# =====================================================

R_B_D01_result = np.zeros(np.size(epsilon_cand))

R_B_D10_result = np.zeros(np.size(epsilon_cand))

P_FA_D01_result = np.zeros(np.size(epsilon_cand))

P_FA_D10_result = np.zeros(np.size(epsilon_cand))

P_MD_D01_result = np.zeros(np.size(epsilon_cand))

P_MD_D10_result = np.zeros(np.size(epsilon_cand))


for ind_epsilon in range(0, np.size(epsilon_cand)):


    # =====================================================
    # Update epsilon and KL threshold
    # =====================================================

    epsilon = epsilon_cand[ind_epsilon]

    sys_param["epsilon"] = epsilon

    sys_param["KL_threshold"] = 2 * epsilon ** 2


    # =====================================================
    # t_max for two KL cases
    # =====================================================

    t_max_D01 = myf_find_t_max_case(
        sys_param,
        "D01"
    )

    t_max_D10 = myf_find_t_max_case(
        sys_param,
        "D10"
    )


    # =====================================================
    # Safety margin for solver tolerance
    # =====================================================

    if "t_max_safety" in sys_param:

        t_max_D01 = sys_param["t_max_safety"] * t_max_D01

        t_max_D10 = sys_param["t_max_safety"] * t_max_D10


    # =====================================================
    # Temporary arrays
    # =====================================================

    R_B_D01_temp = np.zeros(num_samples)

    R_B_D10_temp = np.zeros(num_samples)

    P_FA_D01_temp = np.zeros(num_samples)

    P_FA_D10_temp = np.zeros(num_samples)

    P_MD_D01_temp = np.zeros(num_samples)

    P_MD_D10_temp = np.zeros(num_samples)


    for ind_sample in tqdm(range(0, num_samples)):


        # =====================================================
        # Generate channel
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
        # Robust rate maximization under D(p0 || p1)
        # =====================================================

        sys_param["t_max"] = t_max_D01

        solutions_D01 = myf_algorithm_robust_rate_imperfect_WCSI(
            sys_param,
            channel
        )

        R_B_D01_temp[ind_sample] = myf_Bob_rate_from_Gamma_B(
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
        # Robust rate maximization under D(p1 || p0)
        # =====================================================

        sys_param["t_max"] = t_max_D10

        solutions_D10 = myf_algorithm_robust_rate_imperfect_WCSI(
            sys_param,
            channel
        )

        R_B_D10_temp[ind_sample] = myf_Bob_rate_from_Gamma_B(
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

    R_B_D01_result[ind_epsilon] = np.nanmean(R_B_D01_temp)

    R_B_D10_result[ind_epsilon] = np.nanmean(R_B_D10_temp)

    P_FA_D01_result[ind_epsilon] = np.nanmean(P_FA_D01_temp)

    P_FA_D10_result[ind_epsilon] = np.nanmean(P_FA_D10_temp)

    P_MD_D01_result[ind_epsilon] = np.nanmean(P_MD_D01_temp)

    P_MD_D10_result[ind_epsilon] = np.nanmean(P_MD_D10_temp)


    print("===================================")

    print("epsilon =", epsilon)

    print("KL threshold =", sys_param["KL_threshold"])

    print("t_max_D01 =", t_max_D01)

    print("t_max_D10 =", t_max_D10)

    print("R_B D01 =", R_B_D01_result[ind_epsilon])

    print("R_B D10 =", R_B_D10_result[ind_epsilon])

    print("P_FA D01 =", P_FA_D01_result[ind_epsilon])

    print("P_FA D10 =", P_FA_D10_result[ind_epsilon])

    print("P_MD D01 =", P_MD_D01_result[ind_epsilon])

    print("P_MD D10 =", P_MD_D10_result[ind_epsilon])

    print("===================================")


# =====================================================
# Print final arrays
# =====================================================

print("===================================")

print("epsilon_cand =", epsilon_cand)

print("R_B_D01_result =", R_B_D01_result)

print("R_B_D10_result =", R_B_D10_result)

print("P_FA_D01_result =", P_FA_D01_result)

print("P_FA_D10_result =", P_FA_D10_result)

print("P_MD_D01_result =", P_MD_D01_result)

print("P_MD_D10_result =", P_MD_D10_result)

print("===================================")


# =====================================================
# Plot Fig. 10
# =====================================================

myf_plot_Fig10(
    sys_param,
    epsilon_cand,
    R_B_D01_result,
    R_B_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
)
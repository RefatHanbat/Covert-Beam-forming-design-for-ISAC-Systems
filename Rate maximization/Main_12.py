import numpy as np

from tqdm import tqdm

from fParam import *
from fChannel import *
from fAlgorithm import *
from fCalculations import *
from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 12 settings
# =====================================================

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["v_w"] = 0.001

sys_param["epsilon"] = 0.20

sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2

# Keep same rate-maximization subsection setting
# Use the same value you used in Fig. 10 / Fig. 11
sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1


# =====================================================
# x-axis values for Fig. 12
# =====================================================

N_cand = np.array([4, 5, 6, 7, 8, 9])


# =====================================================
# Number of Monte Carlo samples
# =====================================================

num_samples = 20
# later you can increase to 50 if needed


# =====================================================
# Results
# =====================================================

R_B_D01_result = np.zeros(np.size(N_cand))
R_B_D10_result = np.zeros(np.size(N_cand))


# =====================================================
# t_max values depend only on epsilon
# =====================================================

t_max_D01 = myf_find_t_max_case(sys_param, "D01")
t_max_D10 = myf_find_t_max_case(sys_param, "D10")

if "t_max_safety" in sys_param:

    t_max_D01 = sys_param["t_max_safety"] * t_max_D01
    t_max_D10 = sys_param["t_max_safety"] * t_max_D10


print("===================================")
print("Fig. 12 settings")
print("P_total_dBm =", sys_param["P_total_dBm"])
print("v_w =", sys_param["v_w"])
print("epsilon =", sys_param["epsilon"])
print("KL threshold =", sys_param["KL_threshold"])
print("gamma_MI =", sys_param["gamma_MI"])
print("gamma_SINR =", sys_param["gamma_SINR"])
print("t_max_D01 =", t_max_D01)
print("t_max_D10 =", t_max_D10)
print("===================================")


for ind_N in range(0, np.size(N_cand)):

    sys_param["N"] = int(N_cand[ind_N])

    R_B_D01_temp = np.zeros(num_samples)
    R_B_D10_temp = np.zeros(num_samples)

    for ind_sample in tqdm(range(0, num_samples)):

        # ================================================
        # Generate channel
        # ================================================
        param_channel = {}
        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(sys_param, param_channel)

        # ================================================
        # Solve robust rate maximization for D(p0||p1)
        # ================================================
        sys_param["t_max"] = t_max_D01

        solutions_D01 = myf_algorithm_robust_rate_imperfect_WCSI(
            sys_param,
            channel
        )

        if solutions_D01["W_R_0"] is not None and solutions_D01["W_R_1"] is not None:

            R_B_D01_temp[ind_sample] = myf_Bob_rate_from_Gamma_B(
                sys_param,
                solutions_D01
            )

        else:

            R_B_D01_temp[ind_sample] = np.nan


        # ================================================
        # Solve robust rate maximization for D(p1||p0)
        # ================================================
        sys_param["t_max"] = t_max_D10

        solutions_D10 = myf_algorithm_robust_rate_imperfect_WCSI(
            sys_param,
            channel
        )

        if solutions_D10["W_R_0"] is not None and solutions_D10["W_R_1"] is not None:

            R_B_D10_temp[ind_sample] = myf_Bob_rate_from_Gamma_B(
                sys_param,
                solutions_D10
            )

        else:

            R_B_D10_temp[ind_sample] = np.nan


    # =====================================================
    # Average over Monte Carlo samples
    # =====================================================

    R_B_D01_result[ind_N] = np.nanmean(R_B_D01_temp)
    R_B_D10_result[ind_N] = np.nanmean(R_B_D10_temp)

    print("===================================")
    print("N =", sys_param["N"])
    print("R_B D01 =", R_B_D01_result[ind_N])
    print("R_B D10 =", R_B_D10_result[ind_N])
    print("===================================")


print("===================================")
print("N_cand =", N_cand)
print("R_B_D01_result =", R_B_D01_result)
print("R_B_D10_result =", R_B_D10_result)
print("===================================")


# =====================================================
# Plot Fig. 12
# =====================================================

myf_plot_Fig12(
    sys_param,
    N_cand,
    R_B_D01_result,
    R_B_D10_result
)
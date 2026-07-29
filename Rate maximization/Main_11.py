import numpy as np

from tqdm import tqdm

from fParam import *
from fChannel import *
from fAlgorithm import *
from fCalculations import *
from fPlot import *


sys_param = sys_param


# =====================================================
# Fig. 11 settings
# =====================================================

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["epsilon"] = 0.20

sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2

# Same radar MI threshold used in your rate-maximization subsection
sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1


# x-axis values for Fig. 11
v_w_cand = np.array([
    0.001,
    0.005,
    0.009,
    0.013,
    0.017
])


# =====================================================
# Number of Monte Carlo samples
# =====================================================

num_samples = 20
# later you may increase to 50 or 100


# =====================================================
# Results
# =====================================================

R_B_D01_result = np.zeros(np.size(v_w_cand))
R_B_D10_result = np.zeros(np.size(v_w_cand))

P_FA_D01_result = np.zeros(np.size(v_w_cand))
P_FA_D10_result = np.zeros(np.size(v_w_cand))

P_MD_D01_result = np.zeros(np.size(v_w_cand))
P_MD_D10_result = np.zeros(np.size(v_w_cand))


# =====================================================
# t_max values depend only on KL threshold
# =====================================================

t_max_D01 = myf_find_t_max_case(sys_param, "D01")
t_max_D10 = myf_find_t_max_case(sys_param, "D10")

if "t_max_safety" in sys_param:

    t_max_D01 = sys_param["t_max_safety"] * t_max_D01
    t_max_D10 = sys_param["t_max_safety"] * t_max_D10


print("===================================")
print("Fig. 11 settings")
print("N =", sys_param["N"])
print("P_total_dBm =", sys_param["P_total_dBm"])
print("epsilon =", sys_param["epsilon"])
print("KL threshold =", sys_param["KL_threshold"])
print("gamma_MI =", sys_param["gamma_MI"])
print("gamma_SINR =", sys_param["gamma_SINR"])
print("t_max_D01 =", t_max_D01)
print("t_max_D10 =", t_max_D10)
print("===================================")


for ind_vw in range(0, np.size(v_w_cand)):

    sys_param["v_w"] = v_w_cand[ind_vw]

    R_B_D01_temp = np.zeros(num_samples)
    R_B_D10_temp = np.zeros(num_samples)

    P_FA_D01_temp = np.zeros(num_samples)
    P_FA_D10_temp = np.zeros(num_samples)

    P_MD_D01_temp = np.zeros(num_samples)
    P_MD_D10_temp = np.zeros(num_samples)

    for ind_sample in tqdm(range(0, num_samples)):

        # =================================================
        # Generate channel
        # =================================================
        param_channel = {}
        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(sys_param, param_channel)

        # =================================================
        # Generate true Willie channel
        # =================================================
        param_error = {}
        param_error["seed_seq"] = ind_sample

        h_W_true = myf_true_Willie_channel(
            sys_param,
            channel,
            param_error
        )

        # =================================================
        # Solve robust rate maximization for D(p0||p1)
        # =================================================
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

            P_FA_D01_temp[ind_sample], P_MD_D01_temp[ind_sample] = myf_detection_probability_Willie(
                sys_param,
                h_W_true,
                solutions_D01["W_R_0"],
                solutions_D01["W_R_1"]
            )

        else:

            R_B_D01_temp[ind_sample] = np.nan
            P_FA_D01_temp[ind_sample] = np.nan
            P_MD_D01_temp[ind_sample] = np.nan


        # =================================================
        # Solve robust rate maximization for D(p1||p0)
        # =================================================
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

            P_FA_D10_temp[ind_sample], P_MD_D10_temp[ind_sample] = myf_detection_probability_Willie(
                sys_param,
                h_W_true,
                solutions_D10["W_R_0"],
                solutions_D10["W_R_1"]
            )

        else:

            R_B_D10_temp[ind_sample] = np.nan
            P_FA_D10_temp[ind_sample] = np.nan
            P_MD_D10_temp[ind_sample] = np.nan


    # =====================================================
    # Average over Monte Carlo samples
    # =====================================================

    R_B_D01_result[ind_vw] = np.nanmean(R_B_D01_temp)
    R_B_D10_result[ind_vw] = np.nanmean(R_B_D10_temp)

    P_FA_D01_result[ind_vw] = np.nanmean(P_FA_D01_temp)
    P_FA_D10_result[ind_vw] = np.nanmean(P_FA_D10_temp)

    P_MD_D01_result[ind_vw] = np.nanmean(P_MD_D01_temp)
    P_MD_D10_result[ind_vw] = np.nanmean(P_MD_D10_temp)


    print("===================================")
    print("v_w =", sys_param["v_w"])
    print("epsilon =", sys_param["epsilon"])
    print("R_B D01 =", R_B_D01_result[ind_vw])
    print("R_B D10 =", R_B_D10_result[ind_vw])
    print("P_FA D01 =", P_FA_D01_result[ind_vw])
    print("P_FA D10 =", P_FA_D10_result[ind_vw])
    print("P_MD D01 =", P_MD_D01_result[ind_vw])
    print("P_MD D10 =", P_MD_D10_result[ind_vw])
    print("===================================")


print("===================================")
print("v_w_cand =", v_w_cand)
print("R_B_D01_result =", R_B_D01_result)
print("R_B_D10_result =", R_B_D10_result)
print("P_FA_D01_result =", P_FA_D01_result)
print("P_FA_D10_result =", P_FA_D10_result)
print("P_MD_D01_result =", P_MD_D01_result)
print("P_MD_D10_result =", P_MD_D10_result)
print("===================================")


# =====================================================
# Plot Fig. 11
# =====================================================

myf_plot_Fig11(
    sys_param,
    v_w_cand,
    R_B_D01_result,
    R_B_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
)
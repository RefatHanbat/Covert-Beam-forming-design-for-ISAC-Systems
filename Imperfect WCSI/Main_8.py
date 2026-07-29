import numpy as np

from tqdm import tqdm

from fParam import *
from fChannel import *
from fAlgorithm import *
from fCalculations import *
from fPlot import *

sys_param = sys_param


# =====================================================
# Fig. 8 settings
# =====================================================

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10
sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["v_w"] = 0.001

sys_param["epsilon"] = 0.05
sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2


# beta on x-axis is RATE threshold
beta_rate_cand = np.arange(start=0.5, stop=3.1, step=0.5)


# Monte Carlo sample size
num_samples = 20
# later you can try 50 or 100


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

print("===================================")
print("Fig. 8 settings")
print("N =", sys_param["N"])
print("P_total_dBm =", sys_param["P_total_dBm"])
print("v_w =", sys_param["v_w"])
print("epsilon =", sys_param["epsilon"])
print("KL threshold =", sys_param["KL_threshold"])
print("t_max_D01 =", t_max_D01)
print("t_max_D10 =", t_max_D10)
print("===================================")


# =====================================================
# Result arrays
# =====================================================

MI_D01_result = np.zeros(np.size(beta_rate_cand))
MI_D10_result = np.zeros(np.size(beta_rate_cand))


for ind_beta in range(0, np.size(beta_rate_cand)):

    beta_rate = beta_rate_cand[ind_beta]

    # IMPORTANT:
    # Manuscript beta is RATE threshold
    # But optimization uses SINR threshold
    beta_sinr = 2 ** beta_rate - 1

    MI_D01_temp = np.zeros(num_samples)
    MI_D10_temp = np.zeros(num_samples)

    for ind_sample in tqdm(range(0, num_samples)):

        # update beta for current trial
        sys_param["beta"] = beta_sinr

        param_channel = {}
        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(
            sys_param,
            param_channel
        )

        # ============================================
        # Robust design under D(p0 || p1)
        # ============================================
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

        # ============================================
        # Robust design under D(p1 || p0)
        # ============================================
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

    MI_D01_result[ind_beta] = np.mean(MI_D01_temp)
    MI_D10_result[ind_beta] = np.mean(MI_D10_temp)

    print("===================================")
    print("beta_rate =", beta_rate)
    print("beta_sinr =", beta_sinr)
    print("MI D01 =", MI_D01_result[ind_beta])
    print("MI D10 =", MI_D10_result[ind_beta])
    print("===================================")


print("===================================")
print("beta_rate_cand =", beta_rate_cand)
print("MI_D01_result =", MI_D01_result)
print("MI_D10_result =", MI_D10_result)
print("===================================")


myf_plot_Fig8(
    sys_param,
    beta_rate_cand,
    MI_D01_result,
    MI_D10_result
)
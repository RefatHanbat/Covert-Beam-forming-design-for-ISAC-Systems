import numpy as np

from tqdm import tqdm

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import *

from fPlot import *


sys_param = sys_param

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["v_w"] = 0.001

sys_param["KL_threshold"] = 0.02

######## Radar MI threshold for rate maximization #########

sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1

######## t_max for two KL cases #############


t_max_D01 = myf_find_t_max_case(
    sys_param,
    "D01"
)

t_max_D10 = myf_find_t_max_case(
    sys_param,
    "D10"
)
###### Safety margin for solver tolerance ############

t_max_D01 = sys_param["t_max_safety"] * t_max_D01

t_max_D10 = sys_param["t_max_safety"] * t_max_D10


print("===================================")

print("Fig. 9 settings")

print("N =", sys_param["N"])

print("P_total_dBm =", sys_param["P_total_dBm"])

print("sigma_W_2 =", sys_param["sigma_W_2"])

print("v_w =", sys_param["v_w"])

print("KL threshold =", sys_param["KL_threshold"])

print("gamma_MI =", sys_param["gamma_MI"])

print("gamma_SINR =", sys_param["gamma_SINR"])

print("t_max_D01 =", t_max_D01)

print("t_max_D10 =", t_max_D10)

print("===================================")


num_samples = 50


D01_nonrobust = np.zeros(num_samples)

D10_nonrobust = np.zeros(num_samples)

D01_robust = np.zeros(num_samples)

D10_robust = np.zeros(num_samples)


for ind_sample in tqdm(range(0, num_samples)):

    param_channel = {}

    param_channel["seed_seq"] = ind_sample

    channel = myf_channel(
        sys_param,
        param_channel
    )


    param_error = {}

    param_error["seed_seq"] = ind_sample

    h_W_true = myf_true_Willie_channel(
        sys_param,
        channel,
        param_error
    )


    sys_param["t_max"] = t_max_D01

    solutions_nonrobust_D01 = myf_algorithm_nonrobust_rate_imperfect_WCSI(
        sys_param,
        channel
    )

    D01_nonrobust[ind_sample], _ = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_nonrobust_D01["W_R_0"],
        solutions_nonrobust_D01["W_R_1"]
    )


    solutions_robust_D01 = myf_algorithm_robust_rate_imperfect_WCSI(
        sys_param,
        channel
    )

    D01_robust[ind_sample], _ = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_robust_D01["W_R_0"],
        solutions_robust_D01["W_R_1"]
    )

    sys_param["t_max"] = t_max_D10

    solutions_nonrobust_D10 = myf_algorithm_nonrobust_rate_imperfect_WCSI(
        sys_param,
        channel
    )

    _, D10_nonrobust[ind_sample] = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_nonrobust_D10["W_R_0"],
        solutions_nonrobust_D10["W_R_1"]
    )


    solutions_robust_D10 = myf_algorithm_robust_rate_imperfect_WCSI(
        sys_param,
        channel
    )

    _, D10_robust[ind_sample] = myf_KL_divergence_Willie(
        sys_param,
        h_W_true,
        solutions_robust_D10["W_R_0"],
        solutions_robust_D10["W_R_1"]
    )


KL_threshold = sys_param["KL_threshold"]

print("===================================")

print("D01 nonrobust violation ratio =",
      np.mean(D01_nonrobust > KL_threshold))

print("D01 robust violation ratio =",
      np.mean(D01_robust > KL_threshold))

print("D10 nonrobust violation ratio =",
      np.mean(D10_nonrobust > KL_threshold))

print("D10 robust violation ratio =",
      np.mean(D10_robust > KL_threshold))

print("D01_nonrobust min/max =",
      np.nanmin(D01_nonrobust),
      np.nanmax(D01_nonrobust))

print("D01_robust min/max =",
      np.nanmin(D01_robust),
      np.nanmax(D01_robust))

print("D10_nonrobust min/max =",
      np.nanmin(D10_nonrobust),
      np.nanmax(D10_nonrobust))

print("D10_robust min/max =",
      np.nanmin(D10_robust),
      np.nanmax(D10_robust))

print("===================================")


myf_plot_Fig9_CDF(
    sys_param,
    D01_nonrobust,
    D10_nonrobust,
    D01_robust,
    D10_robust
)
import numpy as np

'''
fParam.py

Refat Khan

'''

def dbm_to_linear(P_dBm):

    P_linear = 10 ** (P_dBm / 10)

    return P_linear


sys_param = {}

sys_param["N"] = 5

sys_param["alpha"] = 1.0

sys_param["theta"] = 0.0

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["KL_threshold"] = 0.005

sys_param["v_w"] = 0.005

sys_param["seed"] = 30245109

sys_param["sigma_1_2"] = 1.0

sys_param["sigma_2_2"] = 1.0


### Noise power ###

sys_param["sigma_B_2_dBm"] = 0

sys_param["sigma_W_2_dBm"] = 0

sys_param["sigma_R_2_dBm"] = 0

sys_param["sigma_B_2"] = dbm_to_linear(sys_param["sigma_B_2_dBm"])

sys_param["sigma_W_2"] = dbm_to_linear(sys_param["sigma_W_2_dBm"])

sys_param["sigma_R_2"] = dbm_to_linear(sys_param["sigma_R_2_dBm"])


### Bob SINR threshold ###

sys_param["beta"] = 1.0


### Bisection parameters ###

sys_param["zeta_1"] = 1e-3

sys_param["I_R_l"] = 0.0


### Robust approximation setting ###

sys_param["num_robust_errors"] = 10


### Number of CDF samples ###
sys_param["num_CDF_samples"] = 1000


############## Figure 09 ##################

sys_param["N"] = 5

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

sys_param["KL_threshold"] = 0.02

sys_param["v_w"] = 0.001

sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1
import numpy as np


'''
fParam.py
Refat Khan

Parameters for rate maximization under imperfect WCSI
Fig. 9 to Fig. 13
'''


def dbm_to_linear(P_dBm):

    P_linear = 10 ** (P_dBm / 10)

    return P_linear


sys_param = {}


# =====================================================
# Basic system parameters
# =====================================================

sys_param["N"] = 5

sys_param["alpha"] = 1.0

sys_param["theta"] = 0.0


# =====================================================
# Power
# =====================================================

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


# =====================================================
# Noise power
# 0 dBm = 1 mW, not zero
# =====================================================

sys_param["sigma_B_2_dBm"] = 0

sys_param["sigma_W_2_dBm"] = 0

sys_param["sigma_R_2_dBm"] = 0

sys_param["sigma_B_2"] = dbm_to_linear(sys_param["sigma_B_2_dBm"])

sys_param["sigma_W_2"] = dbm_to_linear(sys_param["sigma_W_2_dBm"])

sys_param["sigma_R_2"] = dbm_to_linear(sys_param["sigma_R_2_dBm"])


# =====================================================
# Channel variance
# =====================================================

sys_param["sigma_1_2"] = 1.0

sys_param["sigma_2_2"] = 1.0


# =====================================================
# Imperfect Willie CSI
# =====================================================

sys_param["v_w"] = 0.001


# =====================================================
# Covertness threshold
# KL_threshold = 2 epsilon^2
# =====================================================

sys_param["epsilon"] = 0.1

sys_param["KL_threshold"] = 0.02


# =====================================================
# Radar MI threshold for rate maximization
# gamma_MI is in bit
# gamma_SINR = 2^(2 gamma_MI) - 1
# =====================================================

sys_param["gamma_MI"] = 1.0

sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1


# =====================================================
# Bisection parameters
# =====================================================

sys_param["zeta_1"] = 1e-3


# =====================================================
# Random seed
# =====================================================

sys_param["seed"] = 30245109


# =====================================================
# Safety factor for robust KL constraint
# Use 1.0 for pure theory
# Use 0.95 if solver gives small KL violations
# =====================================================

sys_param["t_max_safety"] = 0.95


# =====================================================
# Monte Carlo samples
# =====================================================

sys_param["num_samples"] = 1000
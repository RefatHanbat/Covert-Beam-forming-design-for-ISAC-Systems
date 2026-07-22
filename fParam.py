import numpy as np


'''
fparam.py
Refat Khan
'''

def dbm_to_linear(P_dBm):

    return 10**(P_dBm/10)

sys_param = {}

sys_param["N"] = 5

sys_param["alpha"] = 1

sys_param["theta"] = 0.0

## Noise power ##


sys_param["sigma_B_2_dBm"] = 0

sys_param["sigma_W_2_dBm"] = 0

sys_param["sigma_R_2_dBm"] = 0


#### Convert noise powers to linear scale ##

sys_param["sigma_B_2"] = dbm_to_linear(sys_param["sigma_B_2_dBm"])

sys_param["sigma_W_2"]  = dbm_to_linear(sys_param["sigma_W_2_dBm"])

sys_param["sigma_R_2"] = dbm_to_linear(sys_param["sigma_R_2_dBm"])


#### channel variance ###

sys_param["sigma_1_2"] = 1.0

sys_param["sigma_2_2"] = 1.0

### Total power ###

sys_param["P_total_dBm"] = 10

sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


### Bisection parameters ####

sys_param["zeta_1"]  = 1e-3

sys_param["I_R_l"] = 0.0

sys_param["I_R_end"] = 100.0


sys_param["beta"] = 1.0


#### Solver settings ####


sys_param["solver"] = "MOSEK"

### random seed ##

sys_param["seed"] = 10

###### Bob rate maximization part ####

sys_param["gamma_MI"] = 1.0

#### covert gamma_MI to radar snr threshold ###

sys_param["gamma_SINR"] = 2 **(2*sys_param["gamma_MI"]) - 1


### Radar MI threshold for ZF rate maximization ###
sys_param["gamma_MI"] = 1.0

### Convert radar MI threshold to radar SINR threshold ###
sys_param["gamma_SINR"] = 2 ** (2 * sys_param["gamma_MI"]) - 1


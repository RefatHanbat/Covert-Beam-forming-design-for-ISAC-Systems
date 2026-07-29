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


########### Fig. 6(a): epsilon = 0.01 ##########

sys_param["epsilon"] = 0.01

sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2

v_w_cand = np.array([
    0.1,
    0.2,
    0.3,
    0.4,
    0.5
])


num_samples = 20


MI_D01_result = np.zeros(np.size(v_w_cand))

MI_D10_result = np.zeros(np.size(v_w_cand))


for ind_v in range(0, np.size(v_w_cand)):


    sys_param["v_w"] = v_w_cand[ind_v]


    sys_param["KL_threshold"] = 2 * sys_param["epsilon"] ** 2


    
    ############### Case 1:D(p0 || p1) <= 2 epsilon^2 ####################
    

    t_max_D01 = myf_find_t_max_case(
        sys_param,
        "D01"
    )

    #################### Case 2: D(p1 || p0) <= 2 epsilon^2 ############

    t_max_D10 = myf_find_t_max_case(
        sys_param,
        "D10"
    )

    MI_D01_temp = np.zeros(num_samples)

    MI_D10_temp = np.zeros(num_samples)


    for ind_sample in tqdm(range(0, num_samples)):

        param_channel = {}

        param_channel["seed_seq"] = ind_sample

        channel = myf_channel(
            sys_param,
            param_channel
        )

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

    MI_D01_result[ind_v] = np.nanmean(MI_D01_temp)

    MI_D10_result[ind_v] = np.nanmean(MI_D10_temp)


    print("===================================")

    print("v_w =", sys_param["v_w"])

    print("epsilon =", sys_param["epsilon"])

    print("KL threshold =", sys_param["KL_threshold"])

    print("t_max_D01 =", t_max_D01)

    print("t_max_D10 =", t_max_D10)

    print("MI D01 =", MI_D01_result[ind_v])

    print("MI D10 =", MI_D10_result[ind_v])

    print("===================================")


print("===================================")

print("v_w_cand =", v_w_cand)

print("MI_D01_result =", MI_D01_result)

print("MI_D10_result =", MI_D10_result)

print("===================================")


myf_plot_Fig6a(
    sys_param,
    v_w_cand,
    MI_D01_result,
    MI_D10_result
)
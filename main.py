import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import * 

from fPlot import *

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable: iterable

sys_param = sys_param

print(sys_param)

num_samples = 10

x_axis_name = "P_total_dBm_cand"

if x_axis_name == "P_total_dBm_cand":

    x_axis_cand = np.arange(start = 6, stop = 11, step = 1)

num_algorithms = 4

I_R_result = np.zeros((num_algorithms, np.size(x_axis_cand)))

MI_R_result = np.zeros((num_algorithms, np.size(x_axis_cand)))

R_B_result = np.zeros((num_algorithms, np.size(x_axis_cand)))

for ind1 in tqdm(range(0,num_samples)):

    param_channel = {}

    param_channel["seed_seq"] = ind1

    channel = myf_channel(sys_param,param_channel)

    I_R_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))

    MI_R_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))

    R_B_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))

    for ind2 in range(0, np.size(x_axis_cand)):

        if x_axis_name == "P_total_dBm_cand":

            sys_param["P_total_dBm"] = x_axis_cand[ind2]

            sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])

        #### Algorihtm 1: Covert beamformer based on SDR + bisection ###

        solutions_algorithm_1 = myf_algorithm_1_covert_beamformer(sys_param,channel)

        # print(f"solutions_algorithm_1 : {solutions_algorithm_1}")

        I_R_temp[0, ind2] = solutions_algorithm_1["I_R"]

        MI_R_temp[0,ind2] = myf_radar_MI_from_I_R(sys_param,solutions_algorithm_1)

        ### Algorithm 2: Rate maximization ######

        solutions_algorithm_2 = myf_algorithm_2_rate_maximization(sys_param,channel)

        R_B_temp[1,ind2]  = myf_Bob_rate(sys_param,channel, solutions_algorithm_2)


        ########## Algorithm 3 : ZF beamformer I_R ######


        solutions_algorithm_3 = myf_algorithm_3_ZF_beamformer_IR(sys_param,channel)

        I_R_temp[2,ind2] = solutions_algorithm_3["I_R"]

        MI_R_temp[2,ind2] = myf_radar_MI_from_I_R(sys_param,solutions_algorithm_3)


        ########## Algorithm 4 : ZF beamformer R_B ######

        solutions_algorithm_4 = myf_algorithm_4_ZF_beamformer_RB(sys_param, channel)

        R_B_temp[3, ind2] = myf_Bob_rate(sys_param, channel, solutions_algorithm_4)



    I_R_result = ind1 / (ind1 + 1) * I_R_result + 1 / (ind1 + 1) * I_R_temp

    MI_R_result = ind1/ (ind1 + 1) * MI_R_result + 1 / (ind1 + 1) * MI_R_temp

    R_B_result = ind1 / (ind1 + 1) * R_B_result + 1 / (ind1 + 1) * R_B_temp
    
    # print(MI_R_result)

    # print(R_B_result)


myf_plot_MI_R_and_R_B(
    sys_param,
    x_axis_cand,
    MI_R_result,
    R_B_result,
    x_axis_name
)
        
            

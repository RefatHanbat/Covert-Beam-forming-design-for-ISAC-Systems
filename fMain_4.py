import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fParam import *

from fChannel import *

from fAlgorithm import *

from fCalculations import * 

from fPlot_4 import *


try:

    from tqdm import tqdm

except ModuleNotFoundError:

    tqdm = lambda iterable: iterable


sys_param = sys_param


# =====================================================
# Simulation setting
# =====================================================

num_samples = 1000

x_axis_name = "N_cand"


# =====================================================
# Fig. 3 x-axis:
# Number of antennas N
# P_total fixed at 10 dBm
# =====================================================

if x_axis_name == "N_cand":

    x_axis_cand = np.arange(start=4, stop=15, step=2)

    sys_param["P_total_dBm"] = 10

    sys_param["P_total"] = dbm_to_linear(sys_param["P_total_dBm"])


num_algorithms = 4


# =====================================================
# Result arrays
# Row 0: Covert beamformer I_R
# Row 1: Covert beamformer R_B
# Row 2: ZF beamformer I_R
# Row 3: ZF beamformer R_B
# =====================================================

I_R_result = np.zeros((num_algorithms, np.size(x_axis_cand)))

MI_R_result = np.zeros((num_algorithms, np.size(x_axis_cand)))

R_B_result = np.zeros((num_algorithms, np.size(x_axis_cand)))


for ind1 in tqdm(range(0, num_samples)):


    I_R_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))

    MI_R_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))

    R_B_temp = np.zeros((num_algorithms, np.size(x_axis_cand)))


    for ind2 in range(0, np.size(x_axis_cand)):


        # =====================================================
        # Update number of antennas
        # =====================================================

        if x_axis_name == "N_cand":

            sys_param["N"] = int(x_axis_cand[ind2])


        # =====================================================
        # Important:
        # Since N changes, channel must be generated here
        # =====================================================

        param_channel = {}

        param_channel["seed_seq"] = ind1

        channel = myf_channel(sys_param, param_channel)


        # =====================================================
        # Algorithm 1: Covert beamformer I_R
        # =====================================================

        solutions_algorithm_1 = myf_algorithm_1_covert_beamformer(
            sys_param,
            channel
        )

        I_R_temp[0, ind2] = solutions_algorithm_1["I_R"]

        MI_R_temp[0, ind2] = myf_radar_MI_from_I_R(
            sys_param,
            solutions_algorithm_1
        )


        # =====================================================
        # Algorithm 2: Covert beamformer R_B
        # =====================================================

        solutions_algorithm_2 = myf_algorithm_2_rate_maximization(
            sys_param,
            channel
        )

        R_B_temp[1, ind2] = myf_Bob_rate(
            sys_param,
            channel,
            solutions_algorithm_2
        )


        # =====================================================
        # Algorithm 3: ZF beamformer I_R
        # =====================================================

        solutions_algorithm_3 = myf_algorithm_3_ZF_beamformer_IR(
            sys_param,
            channel
        )

        I_R_temp[2, ind2] = solutions_algorithm_3["I_R"]

        MI_R_temp[2, ind2] = myf_radar_MI_from_I_R(
            sys_param,
            solutions_algorithm_3
        )


        # =====================================================
        # Algorithm 4: ZF beamformer R_B
        # =====================================================

        solutions_algorithm_4 = myf_algorithm_4_ZF_beamformer_RB(
            sys_param,
            channel
        )

        R_B_temp[3, ind2] = myf_Bob_rate(
            sys_param,
            channel,
            solutions_algorithm_4
        )


        # =====================================================
        # Debug print
        # =====================================================

        print(
            "N =", sys_param["N"],
            "| Covert MI =", MI_R_temp[0, ind2],
            "| Covert R_B =", R_B_temp[1, ind2],
            "| ZF MI =", MI_R_temp[2, ind2],
            "| ZF R_B =", R_B_temp[3, ind2]
        )


    # =====================================================
    # Running average
    # =====================================================

    I_R_result = (
        ind1 / (ind1 + 1) * I_R_result
        + 1 / (ind1 + 1) * I_R_temp
    )

    MI_R_result = (
        ind1 / (ind1 + 1) * MI_R_result
        + 1 / (ind1 + 1) * MI_R_temp
    )

    R_B_result = (
        ind1 / (ind1 + 1) * R_B_result
        + 1 / (ind1 + 1) * R_B_temp
    )


# =====================================================
# Plot Fig. 3
# =====================================================

myf_plot_MI_R_and_R_B(
    sys_param,
    x_axis_cand,
    MI_R_result,
    R_B_result,
    x_axis_name
)
import numpy as np


'''
fCalculations.py
Refat Khan
Performance calculation functions for Covert Beamforming IRSC paper
'''


def myf_radar_MI_from_I_R(sys_param, solutions):
    """
    Calculate radar mutual information from I_R.

    Paper notation:

        I(y_R; h_T | s_R)
        = 1/2 log(1 + I_R)

    where I_R is obtained from problem (26).

    Input:
        solutions["I_R"]

    Output:
        radar_MI
    """

    I_R = solutions["I_R"]

    ##### Numerical safety #####

    I_R = np.real(I_R)

    I_R = np.maximum(I_R, 0)

    ###### If you want bits/sec/Hz, use log2 #############

    radar_MI = 0.5 * np.log2(1 + I_R)

    return radar_MI


def myf_Bob_rate(sys_param, channel, solutions):
    """
    Calculate Bob achievable rate.

    Paper notation:

        R_B = log2(1 + SINR_B)

    where

        SINR_B =
        Tr(h_B^H W_R_1 h_B)
        /
        (Tr(h_B^H W_R_0 h_B) + sigma_B^2)

    For rate maximization algorithm, if Gamma_B is already available,
    then:

        R_B = log2(1 + Gamma_B)
    """

    # =========================
    # If Algorithm 2 already gives Gamma_B
    # =========================
    if "Gamma_B" in solutions:

        Gamma_B = solutions["Gamma_B"]

        Gamma_B = np.real(Gamma_B)

        Gamma_B = np.maximum(Gamma_B, 0.0)

        R_B = np.log2(1 + Gamma_B)

        return R_B

    # =========================
    # Otherwise calculate from W_R_0 and W_R_1
    # =========================
    h_B = channel["h_B"]

    W_R_0 = solutions["W_R_0"]

    W_R_1 = solutions["W_R_1"]

    sigma_B_2 = sys_param["sigma_B_2"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan

    Tr_hB_WR1_hB = np.real(
        np.trace(h_B.conj().T @ W_R_1 @ h_B)
    )

    Tr_hB_WR0_hB = np.real(
        np.trace(h_B.conj().T @ W_R_0 @ h_B)
    )

    # Numerical safety
    Tr_hB_WR1_hB = np.maximum(Tr_hB_WR1_hB, 0.0)

    Tr_hB_WR0_hB = np.maximum(Tr_hB_WR0_hB, 0.0)

    SINR_B = Tr_hB_WR1_hB / (Tr_hB_WR0_hB + sigma_B_2)

    R_B = np.log2(1 + SINR_B)

    return R_B
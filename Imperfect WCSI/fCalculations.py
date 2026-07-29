import numpy as np


def myf_KL_from_t(t):

    D_p0_p1 = np.log(1 + t) + 1 / (1 + t) - 1

    D_p1_p0 = -np.log(1 + t) + t

    return D_p0_p1, D_p1_p0


def myf_find_t_max(sys_param):

    """
    Find maximum t such that:

        D(p0||p1) <= KL_threshold
        D(p1||p0) <= KL_threshold

    where:

        t = q / s
    """

    KL_threshold = sys_param["KL_threshold"]

    t_l = 0.0

    t_u = 1.0

    for ind in range(0, 100):

        D01, D10 = myf_KL_from_t(t_u)

        if D01 > KL_threshold or D10 > KL_threshold:

            break

        t_u = 2 * t_u

    for ind in range(0, 100):

        t_mid = (t_l + t_u) / 2

        D01, D10 = myf_KL_from_t(t_mid)

        if D01 <= KL_threshold and D10 <= KL_threshold:

            t_l = t_mid

        else:

            t_u = t_mid

    return t_l


def myf_KL_divergence_Willie(sys_param, h_W_true, W_R_0, W_R_1):

    
    """
    Calculate achieved KL divergence at true Willie channel.

        lambda_0 = h_W^H W_R_0 h_W + sigma_W^2

        lambda_1 = h_W^H W_R_0 h_W
                 + h_W^H W_R_1 h_W
                 + sigma_W^2
    """

    sigma_W_2 = sys_param["sigma_W_2"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan, np.nan

    lambda_0 = np.real(
        np.trace(h_W_true.conj().T @ W_R_0 @ h_W_true)
    ) + sigma_W_2

    lambda_1 = np.real(
        np.trace(h_W_true.conj().T @ W_R_0 @ h_W_true)
    ) + np.real(
        np.trace(h_W_true.conj().T @ W_R_1 @ h_W_true)
    ) + sigma_W_2

    lambda_0 = np.maximum(lambda_0, 1e-12)

    lambda_1 = np.maximum(lambda_1, 1e-12)

    D_p0_p1 = np.log(lambda_1 / lambda_0) + lambda_0 / lambda_1 - 1

    D_p1_p0 = np.log(lambda_0 / lambda_1) + lambda_1 / lambda_0 - 1

    D_p0_p1 = np.real(D_p0_p1)

    D_p1_p0 = np.real(D_p1_p0)

    return D_p0_p1, D_p1_p0


def myf_empirical_CDF(data):

    data = np.array(data)

    data = data[~np.isnan(data)]

    x_sorted = np.sort(data)

    num_data = len(x_sorted)

    cdf = np.arange(1, num_data + 1) / num_data

    return x_sorted, cdf


########## Figure 05 #########################

def myf_find_t_max_case(sys_param, KL_case):

    """
    Find maximum t for one KL constraint.

    KL_case = "D01":
        D(p0 || p1) <= KL_threshold

    KL_case = "D10":
        D(p1 || p0) <= KL_threshold
    """

    KL_threshold = sys_param["KL_threshold"]

    t_l = 0.0

    t_u = 1.0

    for ind in range(0, 100):

        D01, D10 = myf_KL_from_t(t_u)

        if KL_case == "D01":

            if D01 > KL_threshold:

                break

        elif KL_case == "D10":

            if D10 > KL_threshold:

                break

        t_u = 2 * t_u


    for ind in range(0, 100):

        t_mid = (t_l + t_u) / 2

        D01, D10 = myf_KL_from_t(t_mid)

        if KL_case == "D01":

            if D01 <= KL_threshold:

                t_l = t_mid

            else:

                t_u = t_mid

        elif KL_case == "D10":

            if D10 <= KL_threshold:

                t_l = t_mid

            else:

                t_u = t_mid

    return t_l


def myf_detection_probability_Willie(sys_param, h_W_true, W_R_0, W_R_1):
    
    """
    Calculate Willie false alarm and miss detection probabilities.

    False alarm:
        P(D1 | H0)

    Miss detection:
        P(D0 | H1)
    """

    sigma_W_2 = sys_param["sigma_W_2"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan, np.nan

    lambda_0 = np.real(
        np.trace(h_W_true.conj().T @ W_R_0 @ h_W_true)
    ) + sigma_W_2

    lambda_1 = np.real(
        np.trace(h_W_true.conj().T @ W_R_0 @ h_W_true)
    ) + np.real(
        np.trace(h_W_true.conj().T @ W_R_1 @ h_W_true)
    ) + sigma_W_2

    lambda_0 = np.maximum(lambda_0, 1e-12)

    lambda_1 = np.maximum(lambda_1, 1e-12)

    if lambda_1 <= lambda_0 + 1e-12:

        P_FA = np.exp(-1)

        P_MD = 1 - np.exp(-1)

        return P_FA, P_MD

    threshold = (
        lambda_0
        * lambda_1
        / (lambda_1 - lambda_0)
        * np.log(lambda_1 / lambda_0)
    )

    P_FA = np.exp(-threshold / lambda_0)

    P_MD = 1 - np.exp(-threshold / lambda_1)

    P_FA = np.real(P_FA)

    P_MD = np.real(P_MD)

    return P_FA, P_MD

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
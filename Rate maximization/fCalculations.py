import numpy as np


'''
fCalculations.py
Refat Khan

Calculation functions for rate maximization under imperfect WCSI

'''


def myf_KL_from_t(t):

    D_p0_p1 = np.log(1 + t) + 1 / (1 + t) - 1

    D_p1_p0 = -np.log(1 + t) + t

    return D_p0_p1, D_p1_p0


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


def myf_quad_value(h, W):

    """
    Safely calculate h^H W h.
    """

    if W is None:

        return np.nan

    W = (W + W.conj().T) / 2

    value = np.real(
        (h.conj().T @ W @ h).item()
    )

    value = np.maximum(value, 0.0)

    return value


def myf_KL_divergence_Willie(sys_param, h_W_true, W_R_0, W_R_1):

    """
    Calculate KL divergence at Willie.

        D(p0 || p1)
        D(p1 || p0)
    """

    sigma_W_2 = sys_param["sigma_W_2"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan, np.nan

    power_0 = myf_quad_value(
        h_W_true,
        W_R_0
    )

    power_1 = myf_quad_value(
        h_W_true,
        W_R_1
    )

    lambda_0 = power_0 + sigma_W_2

    lambda_1 = power_0 + power_1 + sigma_W_2

    if lambda_0 <= 1e-10 or lambda_1 <= 1e-10:

        return np.nan, np.nan

    D_p0_p1 = np.log(lambda_1 / lambda_0) + lambda_0 / lambda_1 - 1

    D_p1_p0 = np.log(lambda_0 / lambda_1) + lambda_1 / lambda_0 - 1

    D_p0_p1 = np.real(D_p0_p1)

    D_p1_p0 = np.real(D_p1_p0)

    return D_p0_p1, D_p1_p0


def myf_detection_probability_Willie(sys_param, h_W_true, W_R_0, W_R_1):

    """
    Calculate Willie detection probabilities.

        P_FA = P(D1 | H0)
        P_MD = P(D0 | H1)
    """

    sigma_W_2 = sys_param["sigma_W_2"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan, np.nan

    power_0 = myf_quad_value(
        h_W_true,
        W_R_0
    )

    power_1 = myf_quad_value(
        h_W_true,
        W_R_1
    )

    lambda_0 = power_0 + sigma_W_2

    lambda_1 = power_0 + power_1 + sigma_W_2

    if lambda_0 <= 1e-10 or lambda_1 <= 1e-10:

        return np.nan, np.nan

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


def myf_Bob_rate_from_Gamma_B(sys_param, solutions):

    Gamma_B = solutions["Gamma_B"]

    Gamma_B = np.real(Gamma_B)

    Gamma_B = np.maximum(Gamma_B, 0.0)

    R_B = np.log2(1 + Gamma_B)

    return R_B


def myf_radar_MI_from_solution(sys_param, channel, solutions):

    """
    Calculate radar MI from W_R_0 and W_R_1.
    Useful for checking rate-maximization solutions.
    """

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    h_T = channel["h_T"]

    W_R_0 = solutions["W_R_0"]

    W_R_1 = solutions["W_R_1"]

    if W_R_0 is None or W_R_1 is None:

        return np.nan

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    Tr_hT_WR0_hT = myf_quad_value(
        h_T,
        W_R_0
    )

    Tr_hT_WR1_hT = myf_quad_value(
        h_T,
        W_R_1
    )

    I_R = (
        alpha_abs_2
        * Tr_hT_WR0_hT
        * norm_h_T_2
    ) / (
        alpha_abs_2
        * Tr_hT_WR1_hT
        * norm_h_T_2
        + sigma_R_2
    )

    radar_MI = 0.5 * np.log2(1 + I_R)

    return radar_MI


def myf_empirical_CDF(data):

    data = np.array(data)

    data = data[np.isfinite(data)]

    x_sorted = np.sort(data)

    num_data = len(x_sorted)

    cdf = np.arange(1, num_data + 1) / num_data

    return x_sorted, cdf
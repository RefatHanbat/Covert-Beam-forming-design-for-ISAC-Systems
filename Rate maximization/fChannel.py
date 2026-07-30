import numpy as np


'''
fChannel.py
Refat Khan

Channel generation for rate maximization under imperfect WCSI

'''


def myf_steering_vector(sys_param):

    N = sys_param["N"]

    theta = sys_param["theta"]

    n = np.arange(0, N).reshape(N, 1)

    h_T = np.exp(1j * np.pi * n * np.sin(theta))

    return h_T


def myf_complex_gaussian_channel(N, variance):

    h = np.sqrt(variance / 2) * (
        np.random.randn(N, 1)
        + 1j * np.random.randn(N, 1)
    )

    return h


def myf_generate_Willie_error(sys_param, param_error):
    
    """
    Generate bounded Willie CSI error.

        ||Delta h_W||^2 <= v_w

    This is consistent with S-procedure robust design.
    """

    N = sys_param["N"]

    v_w = sys_param["v_w"]

    np.random.seed(
        sys_param["seed"]
        + 100000
        + param_error["seed_seq"]
    )

    error_direction = (
        np.random.randn(N, 1)
        + 1j * np.random.randn(N, 1)
    )

    norm_error_direction = np.linalg.norm(error_direction)

    if norm_error_direction <= 1e-12:

        error_direction = np.ones((N, 1), dtype=complex) / np.sqrt(N)

    else:

        error_direction = error_direction / norm_error_direction

    radius = np.sqrt(v_w) * np.random.rand() ** (1 / (2 * N))

    Delta_h_W = radius * error_direction

    return Delta_h_W


def myf_channel(sys_param, param_channel):

    N = sys_param["N"]

    sigma_1_2 = sys_param["sigma_1_2"]

    sigma_2_2 = sys_param["sigma_2_2"]

    np.random.seed(
        sys_param["seed"]
        + param_channel["seed_seq"]
    )

    channel = {}


    # =====================================================
    # Radar-target channel
    # =====================================================

    h_T = myf_steering_vector(sys_param)

    h_R = h_T.copy()


    # =====================================================
    # Bob channel
    # =====================================================

    h_B = myf_complex_gaussian_channel(
        N,
        sigma_1_2
    )


    # =====================================================
    # Estimated Willie channel
    # =====================================================

    h_W_hat = myf_complex_gaussian_channel(
        N,
        sigma_2_2
    )


    # =====================================================
    # Save channels
    # =====================================================

    channel["h_T"] = h_T

    channel["h_R"] = h_R

    channel["h_B"] = h_B

    channel["h_W_hat"] = h_W_hat.copy()

    # Non-robust design uses estimated Willie CSI
    channel["h_W"] = h_W_hat.copy()

    return channel


def myf_true_Willie_channel(sys_param, channel, param_error):

    h_W_hat = channel["h_W_hat"]

    Delta_h_W = myf_generate_Willie_error(
        sys_param,
        param_error
    )

    h_W_true = h_W_hat + Delta_h_W

    return h_W_true
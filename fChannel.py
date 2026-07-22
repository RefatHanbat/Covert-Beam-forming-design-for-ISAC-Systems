import numpy as np

'''
fChannel.py

Refat Khan
Channel generation for covert beamforming IRSC paper

'''

def myf_steering_vector(sys_param):

    N = sys_param["N"]

    if "theta_rad" in sys_param:

        theta = sys_param["theta_rad"]
    else:

        theta = sys_param["theta"]

    n = np.arange(0,N).reshape(N,1)

    ### ULA steering vector with half-wavelength antenna spacing ###

    h_T = np.exp(1j * np.pi * n * np.sin(theta))

    #### Optional normalization ###

    if "normalize_steering" in sys_param:

        if sys_param["normalize_steering"] == True:

            h_T = h_T / np.sqrt(N)

    return h_T

def myf_complex_gaussian_channel(N,variance):

    h = np.sqrt(variance/2) * (np.random.randn(N,1) + 1j * np.random.randn(N,1))


    return h


def myf_channel(sys_param, param_channel):

    channel = {}

    if "seed_seq" in param_channel:

        np.random.seed(sys_param["seed"] + param_channel["seed_seq"])

    else:

        np.random.seed(sys_param["seed"])

    ### Basic parameters ###

    N = sys_param["N"]

    sigma_1_2 = sys_param["sigma_1_2"]

    sigma_2_2 = sys_param["sigma_2_2"]

    ### Radar-target channel ##

    h_T = myf_steering_vector(sys_param)

    h_R = h_T.copy()

    ### Radar Bob Channel ####

    h_B = myf_complex_gaussian_channel(N,sigma_1_2)

    ### Radar-Willie channel ###

    h_W = myf_complex_gaussian_channel(N,sigma_2_2)

    channel["h_T"] = h_T

    channel["h_R"] = h_R

    channel["h_W"] = h_W

    channel["h_B"] = h_B 

    return channel
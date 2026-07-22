import numpy as np

import cvxpy as cp





'''
fAlgorithm.py
Refat Khan 
'''

def myf_get_I_R_upper_bound(sys_param, channel):
    """
    Dynamic upper bound for I_R bisection.

    From Eq. (26), a loose upper bound is:

        I_R <= |alpha|^2 P_total ||h_T||^4 / sigma_R^2

    This avoids artificial saturation caused by fixed I_R_end.
    """

    alpha = sys_param["alpha"]
    alpha_abs_2 = np.abs(alpha) ** 2

    P_total = sys_param["P_total"]
    sigma_R_2 = sys_param["sigma_R_2"]

    h_T = channel["h_T"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    I_R_upper = alpha_abs_2 * P_total * (norm_h_T_2 ** 2) / sigma_R_2

    # small safety factor
    I_R_upper = 1.2 * I_R_upper

    return I_R_upper

def myf_trace_h_W_h(h,W):

    return cp.real(cp.trace(h.conj().T @ W @ h))


def myf_problem_26_feasibility(sys_param, channel, I_R):

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha)**2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    beta = sys_param["beta"]

    P_total = sys_param["P_total"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W = channel["h_W"]

    norm_h_T_2 = np.linalg.norm(h_T)**2

    ### Optimization variables ###

    W_R_0 = cp.Variable((N,N),hermitian = True)

    W_R_1 = cp.Variable((N,N), hermitian = True)

    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T,W_R_0)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T,W_R_1)

    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B,W_R_0)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B,W_R_1)

    Tr_hW_WR1_hW = myf_trace_h_W_h(h_W,W_R_1)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    Tr_WR1 = cp.real(cp.trace(W_R_1))

    constraints = []

    constraints.append(alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2 >=
                       I_R * (alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2 + sigma_R_2))
    
    constraints.append(Tr_hB_WR1_hB
                       >= 
                       beta * (Tr_hB_WR0_hB + sigma_B_2))
    
    constraints.append(Tr_hW_WR1_hW <= 1e-8)
    
    constraints.append(
        Tr_WR0 + Tr_WR1 <= P_total
    )

    constraints.append(W_R_0 >> 0)

    constraints.append(W_R_1 >> 0)

    problem = cp.Problem(cp.Minimize(0),constraints)

    try:
        problem.solve(
            solver = cp.MOSEK,
            verbose = False
        )

    except Exception:

        problem.solve(

            solver = cp.SCS,

            verbose = False,

            eps = 1e-5,

            max_iters = 20000
        )

    feasible_status = [
        "optimal",
        "optimal_inaccurate"
    ]

    is_feasible = problem.status in feasible_status

    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status

    if is_feasible:

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions






def myf_algorithm_1_covert_beamformer(sys_param,channel):

    ##### Bisection method for problem (26) #####


    #### intialization ####

    # I_R_l = sys_param["I_R_l"]

    # I_R_end = sys_param["I_R_end"]

    # zeta_1 = sys_param["zeta_1"]

    ### dynamic initialization ######

    I_R_l = sys_param["I_R_l"]

    I_R_end = myf_get_I_R_upper_bound(sys_param, channel)
    
    zeta_1 = sys_param["zeta_1"]


    ###### end initialization #####################

    best_solutions = {}

    best_solutions["I_R"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"


    while I_R_end - I_R_l >= zeta_1:

        I_R_mid = (I_R_l + I_R_end) / 2

        solutions_mid = myf_problem_26_feasibility(sys_param, channel,I_R_mid)

        if solutions_mid["feasible"]:

            I_R_l = I_R_mid

            best_solutions["I_R"] = I_R_mid

            best_solutions["W_R_0"] = solutions_mid["W_R_0"]

            best_solutions["W_R_1"] = solutions_mid["W_R_1"]

            best_solutions["status"] = solutions_mid["status"]

        else:

            I_R_end = I_R_mid

    if best_solutions["W_R_0"] is not None:

        eig_WR0 = np.linalg.eigvalsh(best_solutions["W_R_0"])

        eig_WR1 = np.linalg.eigvalsh(best_solutions["W_R_1"])

        best_solutions["rank_W_R_0"] = np.sum(eig_WR0 > 1e-6)

        best_solutions["rank_W_R_1"] = np.sum(eig_WR1 > 1e-6)
            
    else:

        best_solutions["rank_W_R_0"] = None

        best_solutions["rank_W_R_1"] = None

    return best_solutions

def myf_get_Gamma_B_upper_bound(sys_param, channel):

    """
    Dynamic upper bound for Bob SINR bisection.

    Loose upper bound:

        Gamma_B <= P_total ||h_B||^2 / sigma_B^2
    """

    P_total = sys_param["P_total"]

    sigma_B_2 = sys_param["sigma_B_2"]

    h_B = channel["h_B"]

    norm_h_B_2 = np.linalg.norm(h_B) ** 2

    Gamma_B_upper = P_total * norm_h_B_2 / sigma_B_2

    Gamma_B_upper = 1.2 * Gamma_B_upper

    return Gamma_B_upper


def myf_problem_rate_max_feasibility(sys_param, channel, Gamma_B):

    """
    Feasibility check for rate maximization problem.

    For fixed Gamma_B, check whether

        SINR_B >= Gamma_B

    subject to:

        radar MI >= gamma_MI
        perfect covert constraint
        total power constraint
        PSD constraints

    Paper notation:
        W_R_0, W_R_1
        h_T, h_B, h_W
    """

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    P_total = sys_param["P_total"]

    gamma_SINR = sys_param["gamma_SINR"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W = channel["h_W"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    ### Optimization variables ###

    W_R_0 = cp.Variable((N, N), hermitian=True)

    W_R_1 = cp.Variable((N, N), hermitian=True)

    ### Paper notation trace terms ###

    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T, W_R_0)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T, W_R_1)

    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B, W_R_0)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B, W_R_1)

    Tr_hW_WR1_hW = myf_trace_h_W_h(h_W, W_R_1)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    Tr_WR1 = cp.real(cp.trace(W_R_1))

    constraints = []

    # =========================
    # Bob SINR constraint
    # =========================

    constraints.append(
        Tr_hB_WR1_hB
        >=
        Gamma_B * (Tr_hB_WR0_hB + sigma_B_2)
    )

    # =========================
    # Radar MI constraint
    # I(y_R; h_T | s_R) >= gamma_MI
    #
    # Equivalent:
    # radar SINR >= gamma_SINR
    # =========================

    constraints.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        gamma_SINR * (
            alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2
            + sigma_R_2
        )
    )

    # =========================
    # Perfect covert constraint
    # Paper: Tr(h_W^H W_R_1 h_W) = 0
    # Numerically: <= 1e-8
    # =========================
    constraints.append(
        Tr_hW_WR1_hW <= 1e-8
    )

    # =========================
    # Total power constraint
    # =========================
    constraints.append(
        Tr_WR0 + Tr_WR1 <= P_total
    )

    constraints.append(W_R_0 >> 0)

    constraints.append(W_R_1 >> 0)

    problem = cp.Problem(cp.Minimize(0), constraints)

    try:

        problem.solve(
            solver=cp.MOSEK,
            verbose=False
        )

    except Exception:

        problem.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=20000
        )

    feasible_status = [
        "optimal",
        "optimal_inaccurate"
    ]

    is_feasible = problem.status in feasible_status

    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status

    if is_feasible:

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions

def myf_algorithm_2_rate_maximization(sys_param, channel):

    """
    Algorithm 2:
        Rate maximization using SDR + bisection.

    This maximizes Bob SINR Gamma_B.

    After finding Gamma_B, Bob rate is:

        R_B = log2(1 + Gamma_B)

    But for consistency, you can still calculate R_B using myf_Bob_rate()
    from fCalculations.py.
    """

    #### Bisection initialization ####

    Gamma_B_l = 0.0

    Gamma_B_end = myf_get_Gamma_B_upper_bound(sys_param, channel)

    zeta_1 = sys_param["zeta_1"]

    best_solutions = {}

    best_solutions["Gamma_B"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"

    while Gamma_B_end - Gamma_B_l >= zeta_1:

        Gamma_B_mid = (Gamma_B_l + Gamma_B_end) / 2

        solutions_mid = myf_problem_rate_max_feasibility(
            sys_param,
            channel,
            Gamma_B_mid
        )

        if solutions_mid["feasible"]:

            Gamma_B_l = Gamma_B_mid

            best_solutions["Gamma_B"] = Gamma_B_mid

            best_solutions["W_R_0"] = solutions_mid["W_R_0"]

            best_solutions["W_R_1"] = solutions_mid["W_R_1"]

            best_solutions["status"] = solutions_mid["status"]

        else:

            Gamma_B_end = Gamma_B_mid

    if best_solutions["W_R_0"] is not None:

        eig_WR0 = np.linalg.eigvalsh(best_solutions["W_R_0"])

        eig_WR1 = np.linalg.eigvalsh(best_solutions["W_R_1"])

        best_solutions["rank_W_R_0"] = np.sum(eig_WR0 > 1e-6)

        best_solutions["rank_W_R_1"] = np.sum(eig_WR1 > 1e-6)

    else:

        best_solutions["rank_W_R_0"] = None

        best_solutions["rank_W_R_1"] = None

    return best_solutions


###################### Zero Forcing #############################


def myf_algorithm_3_ZF_beamformer_IR(sys_param, channel):

    """
    Algorithm 3:
        Zero-forcing beamformer for radar MI I_R.

    Paper ZF constraints:

        h_T^H w_R_1 = 0
        h_W^H w_R_1 = 0
        h_B^H w_R_0 = 0

    Step 1:
        Minimize power of W_R_1 while satisfying Bob rate/SINR constraint.

    Step 2:
        Use remaining power for W_R_0 to maximize radar sensing gain.

    Output:
        solutions["I_R"]
        solutions["W_R_0"]
        solutions["W_R_1"]
    """

    # =========================
    # Parameters
    # =========================

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    beta = sys_param["beta"]       # beta is SINR threshold here

    P_total = sys_param["P_total"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W = channel["h_W"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    feasible_status = [
        "optimal",
        "optimal_inaccurate"
    ]

    # =====================================================
    # Step 1: Find W_R_1 with minimum power
    # =====================================================

    W_R_1 = cp.Variable((N, N), hermitian=True)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T, W_R_1)

    Tr_hW_WR1_hW = myf_trace_h_W_h(h_W, W_R_1)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B, W_R_1)

    Tr_WR1 = cp.real(cp.trace(W_R_1))

    constraints_W1 = []

    # h_T^H W_R_1 h_T = 0

    constraints_W1.append(

        Tr_hT_WR1_hT <= 1e-8
    )

    # h_W^H W_R_1 h_W = 0

    constraints_W1.append(

        Tr_hW_WR1_hW <= 1e-8
    )

    # Bob SINR requirement when h_B^H w_R_0 = 0:
    # |h_B^H w_R_1|^2 >= beta * sigma_B^2

    constraints_W1.append(

        Tr_hB_WR1_hB >= beta * sigma_B_2
    )

    constraints_W1.append(

        W_R_1 >> 0
    )

    problem_W1 = cp.Problem(

        cp.Minimize(Tr_WR1),

        constraints_W1
    )

    try:

        problem_W1.solve(

            solver=cp.MOSEK,

            verbose=False
        )


    except Exception:

        problem_W1.solve(

            solver=cp.SCS,

            verbose=False,

            eps=1e-5,

            max_iters=20000
        )

    if problem_W1.status not in feasible_status:

        solutions = {}

        solutions["I_R"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

        solutions["P_R"] = None

        solutions["P_remaining"] = None

        solutions["status"] = "ZF_W1_infeasible"

        solutions["rank_W_R_0"] = None

        solutions["rank_W_R_1"] = None


        return solutions

    W_R_1_value = W_R_1.value

    P_R = np.real(
        np.trace(W_R_1_value)
    )

    P_R = np.maximum(P_R, 0.0)

    # =====================================================
    # Step 2: Optimize W_R_0 using remaining power
    # =====================================================

    P_remaining = P_total - P_R

    if P_remaining <= 0:

        solutions = {}

        solutions["I_R"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = W_R_1_value

        solutions["P_R"] = P_R

        solutions["P_remaining"] = P_remaining

        solutions["status"] = "ZF_no_remaining_power"

        solutions["rank_W_R_0"] = None

        solutions["rank_W_R_1"] = np.sum(
            
            np.linalg.eigvalsh(W_R_1_value) > 1e-6
        )

        return solutions

    W_R_0 = cp.Variable((N, N), hermitian=True)

    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T, W_R_0)
    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B, W_R_0)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    constraints_W0 = []

    # h_B^H W_R_0 h_B = 0
    constraints_W0.append(
        Tr_hB_WR0_hB <= 1e-8
    )

    # Remaining power constraint
    constraints_W0.append(
        Tr_WR0 <= P_remaining
    )

    constraints_W0.append(
        W_R_0 >> 0
    )

    problem_W0 = cp.Problem(
        cp.Maximize(Tr_hT_WR0_hT),
        constraints_W0
    )

    try:

        problem_W0.solve(
            solver=cp.MOSEK,
            verbose=False
        )

    except Exception:

        problem_W0.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=20000
        )

    if problem_W0.status not in feasible_status:

        solutions = {}

        solutions["I_R"] = 0.0
        solutions["W_R_0"] = None
        solutions["W_R_1"] = W_R_1_value
        solutions["P_R"] = P_R
        solutions["P_remaining"] = P_remaining
        solutions["status"] = "ZF_W0_infeasible"
        solutions["rank_W_R_0"] = None
        solutions["rank_W_R_1"] = np.sum(
            np.linalg.eigvalsh(W_R_1_value) > 1e-6
        )

        return solutions

    W_R_0_value = W_R_0.value

    # =====================================================
    # Calculate I_R for ZF beamformer
    # =====================================================

    Tr_hT_WR0_hT_value = np.real(
        np.trace(h_T.conj().T @ W_R_0_value @ h_T)
    )

    Tr_hT_WR1_hT_value = np.real(
        np.trace(h_T.conj().T @ W_R_1_value @ h_T)
    )

    Tr_hT_WR0_hT_value = np.maximum(
        Tr_hT_WR0_hT_value,
        0.0
    )

    Tr_hT_WR1_hT_value = np.maximum(
        Tr_hT_WR1_hT_value,
        0.0
    )

    numerator = (
        alpha_abs_2
        * Tr_hT_WR0_hT_value
        * norm_h_T_2
    )

    denominator = (
        alpha_abs_2
        * Tr_hT_WR1_hT_value
        * norm_h_T_2
        + sigma_R_2
    )

    I_R = numerator / denominator

    I_R = np.real(I_R)

    I_R = np.maximum(I_R, 0.0)

    # =====================================================
    # Save solutions
    # =====================================================

    solutions = {}

    solutions["I_R"] = I_R

    solutions["W_R_0"] = W_R_0_value

    solutions["W_R_1"] = W_R_1_value

    solutions["P_R"] = P_R

    solutions["P_remaining"] = P_remaining

    solutions["status"] = "optimal"

    eig_WR0 = np.linalg.eigvalsh(W_R_0_value)

    eig_WR1 = np.linalg.eigvalsh(W_R_1_value)

    solutions["rank_W_R_0"] = np.sum(eig_WR0 > 1e-6)

    solutions["rank_W_R_1"] = np.sum(eig_WR1 > 1e-6)

    return solutions



def myf_algorithm_4_ZF_beamformer_RB(sys_param, channel):
    """
    Algorithm 4:
        Zero-forcing beamformer for Bob rate R_B.

    Paper ZF constraints:

        h_T^H w_R_1 = 0
        h_W^H w_R_1 = 0
        h_B^H w_R_0 = 0

    Step 1:
        Minimize power of W_R_0 while satisfying radar MI constraint.

    Step 2:
        Use remaining power for W_R_1 to maximize Bob rate.

    Output:
        solutions["Gamma_B"]
        solutions["W_R_0"]
        solutions["W_R_1"]
    """

    # =========================
    # Parameters
    # =========================
    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    P_total = sys_param["P_total"]

    gamma_SINR = sys_param["gamma_SINR"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W = channel["h_W"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    feasible_status = [
        "optimal",
        "optimal_inaccurate"
    ]

    # =====================================================
    # Step 1: Find W_R_0 with minimum power
    # =====================================================

    W_R_0 = cp.Variable((N, N), hermitian=True)

    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T, W_R_0)

    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B, W_R_0)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    constraints_W0 = []

    # Radar MI requirement:
    #
    # Since h_T^H W_R_1 h_T = 0 under ZF,
    #
    # alpha^2 Tr(h_T^H W_R_0 h_T)||h_T||^2 / sigma_R^2
    # >= gamma_SINR
    constraints_W0.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        gamma_SINR * sigma_R_2
    )

    # h_B^H W_R_0 h_B = 0
    constraints_W0.append(
        Tr_hB_WR0_hB <= 1e-8
    )

    # Total power safety
    constraints_W0.append(
        Tr_WR0 <= P_total
    )

    constraints_W0.append(
        W_R_0 >> 0
    )

    problem_W0 = cp.Problem(
        cp.Minimize(Tr_WR0),
        constraints_W0
    )

    try:

        problem_W0.solve(
            solver=cp.MOSEK,
            verbose=False
        )

    except Exception:

        problem_W0.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=20000
        )

    if problem_W0.status not in feasible_status:

        solutions = {}

        solutions["Gamma_B"] = 0.0
        solutions["I_R"] = 0.0
        solutions["W_R_0"] = None
        solutions["W_R_1"] = None
        solutions["P_R0"] = None
        solutions["P_remaining"] = None
        solutions["status"] = "ZF_RB_W0_infeasible"
        solutions["rank_W_R_0"] = None
        solutions["rank_W_R_1"] = None

        return solutions

    W_R_0_value = W_R_0.value

    P_R0 = np.real(
        np.trace(W_R_0_value)
    )

    P_R0 = np.maximum(P_R0, 0.0)

    # =====================================================
    # Step 2: Optimize W_R_1 using remaining power
    # =====================================================

    P_remaining = P_total - P_R0

    if P_remaining <= 0:

        solutions = {}

        solutions["Gamma_B"] = 0.0
        solutions["I_R"] = 0.0
        solutions["W_R_0"] = W_R_0_value
        solutions["W_R_1"] = None
        solutions["P_R0"] = P_R0
        solutions["P_remaining"] = P_remaining
        solutions["status"] = "ZF_RB_no_remaining_power"
        solutions["rank_W_R_0"] = np.sum(
            np.linalg.eigvalsh(W_R_0_value) > 1e-6
        )
        solutions["rank_W_R_1"] = None

        return solutions

    W_R_1 = cp.Variable((N, N), hermitian=True)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T, W_R_1)

    Tr_hW_WR1_hW = myf_trace_h_W_h(h_W, W_R_1)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B, W_R_1)

    Tr_WR1 = cp.real(cp.trace(W_R_1))

    constraints_W1 = []

    # h_T^H W_R_1 h_T = 0
    constraints_W1.append(
        Tr_hT_WR1_hT <= 1e-8
    )

    # h_W^H W_R_1 h_W = 0
    constraints_W1.append(
        Tr_hW_WR1_hW <= 1e-8
    )

    # Remaining power constraint
    constraints_W1.append(
        Tr_WR1 <= P_remaining
    )

    constraints_W1.append(
        W_R_1 >> 0
    )

    problem_W1 = cp.Problem(
        cp.Maximize(Tr_hB_WR1_hB),
        constraints_W1
    )

    try:

        problem_W1.solve(
            solver=cp.MOSEK,
            verbose=False
        )

    except Exception:

        problem_W1.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-5,
            max_iters=20000
        )

    if problem_W1.status not in feasible_status:

        solutions = {}

        solutions["Gamma_B"] = 0.0
        solutions["I_R"] = 0.0
        solutions["W_R_0"] = W_R_0_value
        solutions["W_R_1"] = None
        solutions["P_R0"] = P_R0
        solutions["P_remaining"] = P_remaining
        solutions["status"] = "ZF_RB_W1_infeasible"
        solutions["rank_W_R_0"] = np.sum(
            np.linalg.eigvalsh(W_R_0_value) > 1e-6
        )
        solutions["rank_W_R_1"] = None

        return solutions

    W_R_1_value = W_R_1.value

    # =====================================================
    # Calculate Gamma_B and I_R
    # =====================================================

    Tr_hB_WR1_hB_value = np.real(
        np.trace(h_B.conj().T @ W_R_1_value @ h_B)
    )

    Tr_hB_WR0_hB_value = np.real(
        np.trace(h_B.conj().T @ W_R_0_value @ h_B)
    )

    Tr_hT_WR0_hT_value = np.real(
        np.trace(h_T.conj().T @ W_R_0_value @ h_T)
    )

    Tr_hT_WR1_hT_value = np.real(
        np.trace(h_T.conj().T @ W_R_1_value @ h_T)
    )

    Tr_hB_WR1_hB_value = np.maximum(
        Tr_hB_WR1_hB_value,
        0.0
    )

    Tr_hB_WR0_hB_value = np.maximum(
        Tr_hB_WR0_hB_value,
        0.0
    )

    Tr_hT_WR0_hT_value = np.maximum(
        Tr_hT_WR0_hT_value,
        0.0
    )

    Tr_hT_WR1_hT_value = np.maximum(
        Tr_hT_WR1_hT_value,
        0.0
    )

    Gamma_B = Tr_hB_WR1_hB_value / (
        Tr_hB_WR0_hB_value + sigma_B_2
    )

    numerator_IR = (
        alpha_abs_2
        * Tr_hT_WR0_hT_value
        * norm_h_T_2
    )

    denominator_IR = (
        alpha_abs_2
        * Tr_hT_WR1_hT_value
        * norm_h_T_2
        + sigma_R_2
    )

    I_R = numerator_IR / denominator_IR

    Gamma_B = np.real(Gamma_B)

    Gamma_B = np.maximum(Gamma_B, 0.0)

    I_R = np.real(I_R)

    I_R = np.maximum(I_R, 0.0)

    # =====================================================
    # Save solutions
    # =====================================================

    solutions = {}

    solutions["Gamma_B"] = Gamma_B

    solutions["I_R"] = I_R

    solutions["W_R_0"] = W_R_0_value

    solutions["W_R_1"] = W_R_1_value

    solutions["P_R0"] = P_R0

    solutions["P_remaining"] = P_remaining

    solutions["status"] = "optimal"

    eig_WR0 = np.linalg.eigvalsh(W_R_0_value)

    eig_WR1 = np.linalg.eigvalsh(W_R_1_value)

    solutions["rank_W_R_0"] = np.sum(eig_WR0 > 1e-6)

    solutions["rank_W_R_1"] = np.sum(eig_WR1 > 1e-6)

    return solutions
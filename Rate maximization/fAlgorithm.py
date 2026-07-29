import numpy as np
import cvxpy as cp


'''
fAlgorithm.py
Refat Khan

Rate maximization algorithms under imperfect WCSI
Robust and non-robust designs
'''


def myf_trace_h_W_h(h, W):

    return cp.real(cp.trace(h.conj().T @ W @ h))


def myf_get_Gamma_B_upper_bound(sys_param, channel):

    P_total = sys_param["P_total"]

    sigma_B_2 = sys_param["sigma_B_2"]

    h_B = channel["h_B"]

    norm_h_B_2 = np.linalg.norm(h_B) ** 2

    Gamma_B_upper = P_total * norm_h_B_2 / sigma_B_2

    Gamma_B_upper = 1.2 * Gamma_B_upper

    return Gamma_B_upper


def myf_add_robust_covertness_constraint(
    constraints,
    sys_param,
    h_W_hat,
    W_R_0,
    W_R_1,
    t_max
):
    """
    Robust covertness constraint using S-procedure.

    h_W = h_W_hat + Delta h_W

    ||Delta h_W||^2 <= v_w
    """

    N = sys_param["N"]

    v_w = sys_param["v_w"]

    sigma_W_2 = sys_param["sigma_W_2"]

    A = W_R_1 - t_max * W_R_0

    rho = cp.Variable(nonneg=True)

    I_N = np.eye(N)

    upper_left = rho * I_N - A

    upper_right = -A @ h_W_hat

    lower_left = upper_right.conj().T

    lower_right = (
        t_max * sigma_W_2
        - cp.real(cp.trace(h_W_hat.conj().T @ A @ h_W_hat))
        - rho * v_w
    )

    lower_right = cp.reshape(
        lower_right,
        (1, 1),
        order="C"
    )

    LMI_matrix = cp.bmat(
        [
            [upper_left, upper_right],
            [lower_left, lower_right]
        ]
    )

    constraints.append(
        LMI_matrix >> 0
    )

    return constraints


def myf_problem_nonrobust_rate_imperfect_WCSI_feasibility(
    sys_param,
    channel,
    Gamma_B
):
    """
    Non-robust rate maximization feasibility problem.

    Uses h_W_hat only.
    """

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    sigma_W_2 = sys_param["sigma_W_2"]

    P_total = sys_param["P_total"]

    gamma_SINR = sys_param["gamma_SINR"]

    t_max = sys_param["t_max"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W_hat = channel["h_W_hat"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2


    W_R_0 = cp.Variable((N, N), hermitian=True)

    W_R_1 = cp.Variable((N, N), hermitian=True)


    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T, W_R_0)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T, W_R_1)

    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B, W_R_0)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B, W_R_1)

    Tr_hW_WR0_hW = myf_trace_h_W_h(h_W_hat, W_R_0)

    Tr_hW_WR1_hW = myf_trace_h_W_h(h_W_hat, W_R_1)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    Tr_WR1 = cp.real(cp.trace(W_R_1))


    constraints = []


    # Bob SINR constraint
    constraints.append(
        Tr_hB_WR1_hB
        >=
        Gamma_B * (Tr_hB_WR0_hB + sigma_B_2)
    )


    # Radar MI constraint
    constraints.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        gamma_SINR * (
            alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2
            + sigma_R_2
        )
    )


    # Non-robust covertness constraint
    constraints.append(
        Tr_hW_WR1_hW
        <=
        t_max * (Tr_hW_WR0_hW + sigma_W_2)
    )


    # Power and PSD constraints
    constraints.append(
        Tr_WR0 + Tr_WR1 <= P_total
    )

    constraints.append(W_R_0 >> 0)

    constraints.append(W_R_1 >> 0)


    problem = cp.Problem(
        cp.Minimize(0),
        constraints
    )


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
        "optimal"
    ]

    is_feasible = problem.status in feasible_status


    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status


    if is_feasible:

        solutions["Gamma_B"] = Gamma_B

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["Gamma_B"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions


def myf_problem_robust_rate_imperfect_WCSI_feasibility(
    sys_param,
    channel,
    Gamma_B
):
    """
    Robust rate maximization feasibility problem.

    Uses S-procedure robust covertness constraint.
    """

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    P_total = sys_param["P_total"]

    gamma_SINR = sys_param["gamma_SINR"]

    t_max = sys_param["t_max"]

    h_T = channel["h_T"]

    h_B = channel["h_B"]

    h_W_hat = channel["h_W_hat"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2


    W_R_0 = cp.Variable((N, N), hermitian=True)

    W_R_1 = cp.Variable((N, N), hermitian=True)


    Tr_hT_WR0_hT = myf_trace_h_W_h(h_T, W_R_0)

    Tr_hT_WR1_hT = myf_trace_h_W_h(h_T, W_R_1)

    Tr_hB_WR0_hB = myf_trace_h_W_h(h_B, W_R_0)

    Tr_hB_WR1_hB = myf_trace_h_W_h(h_B, W_R_1)

    Tr_WR0 = cp.real(cp.trace(W_R_0))

    Tr_WR1 = cp.real(cp.trace(W_R_1))


    constraints = []


    # Bob SINR constraint
    constraints.append(
        Tr_hB_WR1_hB
        >=
        Gamma_B * (Tr_hB_WR0_hB + sigma_B_2)
    )


    # Radar MI constraint
    constraints.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        gamma_SINR * (
            alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2
            + sigma_R_2
        )
    )


    # Robust covertness constraint
    constraints = myf_add_robust_covertness_constraint(
        constraints,
        sys_param,
        h_W_hat,
        W_R_0,
        W_R_1,
        t_max
    )


    # Power and PSD constraints
    constraints.append(
        Tr_WR0 + Tr_WR1 <= P_total
    )

    constraints.append(W_R_0 >> 0)

    constraints.append(W_R_1 >> 0)


    problem = cp.Problem(
        cp.Minimize(0),
        constraints
    )


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
        "optimal"
    ]

    is_feasible = problem.status in feasible_status


    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status


    if is_feasible:

        solutions["Gamma_B"] = Gamma_B

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["Gamma_B"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions


def myf_algorithm_nonrobust_rate_imperfect_WCSI(sys_param, channel):

    Gamma_B_l = 0.0

    Gamma_B_end = myf_get_Gamma_B_upper_bound(
        sys_param,
        channel
    )

    zeta_1 = sys_param["zeta_1"]


    best_solutions = {}

    best_solutions["Gamma_B"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"


    while Gamma_B_end - Gamma_B_l >= zeta_1:

        Gamma_B_mid = (Gamma_B_l + Gamma_B_end) / 2

        solutions_mid = myf_problem_nonrobust_rate_imperfect_WCSI_feasibility(
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

    return best_solutions


def myf_algorithm_robust_rate_imperfect_WCSI(sys_param, channel):

    Gamma_B_l = 0.0

    Gamma_B_end = myf_get_Gamma_B_upper_bound(
        sys_param,
        channel
    )

    zeta_1 = sys_param["zeta_1"]


    best_solutions = {}

    best_solutions["Gamma_B"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"


    while Gamma_B_end - Gamma_B_l >= zeta_1:

        Gamma_B_mid = (Gamma_B_l + Gamma_B_end) / 2

        solutions_mid = myf_problem_robust_rate_imperfect_WCSI_feasibility(
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

    return best_solutions
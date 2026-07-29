import numpy as np
import cvxpy as cp


def myf_trace_h_W_h(h, W):

    return cp.real(cp.trace(h.conj().T @ W @ h))


def myf_get_I_R_upper_bound(sys_param, channel):

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    P_total = sys_param["P_total"]

    sigma_R_2 = sys_param["sigma_R_2"]

    h_T = channel["h_T"]

    norm_h_T_2 = np.linalg.norm(h_T) ** 2

    I_R_upper = alpha_abs_2 * P_total * (norm_h_T_2 ** 2) / sigma_R_2

    I_R_upper = 1.2 * I_R_upper

    return I_R_upper


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

    Channel uncertainty:

        h_W = h_W_hat + Delta h_W

        ||Delta h_W||^2 <= v_w

    Covertness constraint:

        h_W^H W_R_1 h_W
        <=
        t_max * (h_W^H W_R_0 h_W + sigma_W^2)

    Equivalent:

        h_W^H A h_W <= t_max * sigma_W^2

    where:

        A = W_R_1 - t_max * W_R_0
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


def myf_problem_nonrobust_imperfect_WCSI_feasibility(
    sys_param,
    channel,
    I_R
):
    """
    Non-robust imperfect-WCSI feasibility problem.

    It designs beamformer only using estimated Willie channel:

        h_W_hat

    Constraint:

        h_W_hat^H W_R_1 h_W_hat
        <=
        t_max * (h_W_hat^H W_R_0 h_W_hat + sigma_W^2)

    This is non-robust because it does not protect against Delta h_W.
    """

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    sigma_W_2 = sys_param["sigma_W_2"]

    beta = sys_param["beta"]

    P_total = sys_param["P_total"]

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


    # =====================================================
    # Radar MI/SINR constraint
    # =====================================================

    constraints.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        I_R * (
            alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2
            + sigma_R_2
        )
    )


    # =====================================================
    # Bob SINR constraint
    # =====================================================

    constraints.append(
        Tr_hB_WR1_hB
        >=
        beta * (Tr_hB_WR0_hB + sigma_B_2)
    )


    # =====================================================
    # Non-robust KL-based covertness constraint
    # only for h_W_hat
    # =====================================================

    constraints.append(
        Tr_hW_WR1_hW
        <=
        t_max * (Tr_hW_WR0_hW + sigma_W_2)
    )


    # =====================================================
    # Power and PSD constraints
    # =====================================================

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
        "optimal",
        "optimal_inaccurate"
    ]

    is_feasible = problem.status in feasible_status


    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status

    if is_feasible:

        solutions["I_R"] = I_R

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["I_R"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions


def myf_problem_robust_imperfect_WCSI_feasibility(
    sys_param,
    channel,
    I_R
):
    """
    Robust imperfect-WCSI feasibility problem.

    It protects the covertness constraint for all Willie channel errors:

        h_W = h_W_hat + Delta h_W

        ||Delta h_W||^2 <= v_w

    using S-procedure.
    """

    N = sys_param["N"]

    alpha = sys_param["alpha"]

    alpha_abs_2 = np.abs(alpha) ** 2

    sigma_R_2 = sys_param["sigma_R_2"]

    sigma_B_2 = sys_param["sigma_B_2"]

    beta = sys_param["beta"]

    P_total = sys_param["P_total"]

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


    # =====================================================
    # Radar MI/SINR constraint
    # =====================================================

    constraints.append(
        alpha_abs_2 * Tr_hT_WR0_hT * norm_h_T_2
        >=
        I_R * (
            alpha_abs_2 * Tr_hT_WR1_hT * norm_h_T_2
            + sigma_R_2
        )
    )


    # =====================================================
    # Bob SINR constraint
    # =====================================================

    constraints.append(
        Tr_hB_WR1_hB
        >=
        beta * (Tr_hB_WR0_hB + sigma_B_2)
    )


    # =====================================================
    # Robust KL-based covertness constraint
    # using S-procedure
    # =====================================================

    constraints = myf_add_robust_covertness_constraint(
        constraints,
        sys_param,
        h_W_hat,
        W_R_0,
        W_R_1,
        t_max
    )


    # =====================================================
    # Power and PSD constraints
    # =====================================================

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
        "optimal",
        "optimal_inaccurate"
    ]

    is_feasible = problem.status in feasible_status


    solutions = {}

    solutions["feasible"] = is_feasible

    solutions["status"] = problem.status

    if is_feasible:

        solutions["I_R"] = I_R

        solutions["W_R_0"] = W_R_0.value

        solutions["W_R_1"] = W_R_1.value

    else:

        solutions["I_R"] = 0.0

        solutions["W_R_0"] = None

        solutions["W_R_1"] = None

    return solutions


def myf_algorithm_nonrobust_imperfect_WCSI(sys_param, channel):
    """
    Non-robust imperfect-WCSI design.

    Uses only h_W_hat in the covertness constraint.
    """

    I_R_l = sys_param["I_R_l"]

    I_R_end = myf_get_I_R_upper_bound(
        sys_param,
        channel
    )

    zeta_1 = sys_param["zeta_1"]


    best_solutions = {}

    best_solutions["I_R"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"


    while I_R_end - I_R_l >= zeta_1:

        I_R_mid = (I_R_l + I_R_end) / 2

        solutions_mid = myf_problem_nonrobust_imperfect_WCSI_feasibility(
            sys_param,
            channel,
            I_R_mid
        )

        if solutions_mid["feasible"]:

            I_R_l = I_R_mid

            best_solutions["I_R"] = I_R_mid

            best_solutions["W_R_0"] = solutions_mid["W_R_0"]

            best_solutions["W_R_1"] = solutions_mid["W_R_1"]

            best_solutions["status"] = solutions_mid["status"]

        else:

            I_R_end = I_R_mid

    return best_solutions


def myf_algorithm_robust_imperfect_WCSI(sys_param, channel, param_robust=None):
    """
    Robust imperfect-WCSI design.

    Uses S-procedure robust covertness constraint.

    param_robust is kept only so your existing main.py does not break.
    It is not needed anymore.
    """

    I_R_l = sys_param["I_R_l"]

    I_R_end = myf_get_I_R_upper_bound(
        sys_param,
        channel
    )

    zeta_1 = sys_param["zeta_1"]


    best_solutions = {}

    best_solutions["I_R"] = 0.0

    best_solutions["W_R_0"] = None

    best_solutions["W_R_1"] = None

    best_solutions["status"] = "not_solved"


    while I_R_end - I_R_l >= zeta_1:

        I_R_mid = (I_R_l + I_R_end) / 2

        solutions_mid = myf_problem_robust_imperfect_WCSI_feasibility(
            sys_param,
            channel,
            I_R_mid
        )

        if solutions_mid["feasible"]:

            I_R_l = I_R_mid

            best_solutions["I_R"] = I_R_mid

            best_solutions["W_R_0"] = solutions_mid["W_R_0"]

            best_solutions["W_R_1"] = solutions_mid["W_R_1"]

            best_solutions["status"] = solutions_mid["status"]

        else:

            I_R_end = I_R_mid

    return best_solutions


########## From figure 9 to figure 13 ##############


def myf_get_Gamma_B_upper_bound(sys_param, channel):

    P_total = sys_param["P_total"]

    sigma_B_2 = sys_param["sigma_B_2"]

    h_B = channel["h_B"]

    norm_h_B_2 = np.linalg.norm(h_B) ** 2

    Gamma_B_upper = P_total * norm_h_B_2 / sigma_B_2

    Gamma_B_upper = 1.2 * Gamma_B_upper

    return Gamma_B_upper


def myf_problem_nonrobust_rate_imperfect_WCSI_feasibility(
    sys_param,
    channel,
    Gamma_B
):
    """
    Non-robust rate maximization feasibility problem.

    It uses only estimated Willie channel h_W_hat.
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


    # Non-robust covertness constraint using h_W_hat only
    constraints.append(
        Tr_hW_WR1_hW
        <=
        t_max * (Tr_hW_WR0_hW + sigma_W_2)
    )


    # Total power constraint
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
        "optimal",
        "optimal_inaccurate"
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

    It uses S-procedure robust covertness constraint.
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


    # Robust covertness constraint using S-procedure
    constraints = myf_add_robust_covertness_constraint(
        constraints,
        sys_param,
        h_W_hat,
        W_R_0,
        W_R_1,
        t_max
    )


    # Total power constraint
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
        "optimal",
        "optimal_inaccurate"
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

    """
    Non-robust rate maximization under imperfect WCSI.
    """

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

    """
    Robust rate maximization under imperfect WCSI.
    """

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
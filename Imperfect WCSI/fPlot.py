import numpy as np

import matplotlib.pyplot as plt

import os


def myf_empirical_CDF(data):
    """
    Generate empirical CDF.
    """

    data = np.array(data)

    data = data[~np.isnan(data)]

    x_sorted = np.sort(data)

    num_data = len(x_sorted)

    cdf = np.arange(1, num_data + 1) / num_data

    return x_sorted, cdf


def myf_plot_Fig4_CDF(
    sys_param,
    D01_nonrobust,
    D10_nonrobust,
    D01_robust,
    D10_robust
):

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    KL_threshold = sys_param["KL_threshold"]


    # =====================================================
    # Fig. 4(a): D(p0 || p1)
    # =====================================================

    x_D01_nonrobust, cdf_D01_nonrobust = myf_empirical_CDF(
        D01_nonrobust
    )

    x_D01_robust, cdf_D01_robust = myf_empirical_CDF(
        D01_robust
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        x_D01_nonrobust,
        cdf_D01_nonrobust,
        color="blue",
        linewidth=2.5,
        linestyle="-",
        label="Non-robust design"
    )

    plt.plot(
        x_D01_robust,
        cdf_D01_robust,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label="Robust design"
    )

    plt.axvline(
        KL_threshold,
        color="black",
        linewidth=2,
        linestyle=":",
        label=r"Threshold $2\epsilon^2$"
    )

    plt.xlabel(r"$D(p_0||p_1)$", fontsize=14)

    plt.ylabel("Empirical CDF", fontsize=14)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.7
    )

    plt.legend(fontsize=11)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig4a_CDF_D_p0_p1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # Fig. 4(b): D(p1 || p0)
    # =====================================================

    x_D10_nonrobust, cdf_D10_nonrobust = myf_empirical_CDF(
        D10_nonrobust
    )

    x_D10_robust, cdf_D10_robust = myf_empirical_CDF(
        D10_robust
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        x_D10_nonrobust,
        cdf_D10_nonrobust,
        color="blue",
        linewidth=2.5,
        linestyle="-",
        label="Non-robust design"
    )

    plt.plot(
        x_D10_robust,
        cdf_D10_robust,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label="Robust design"
    )

    plt.axvline(
        KL_threshold,
        color="black",
        linewidth=2,
        linestyle=":",
        label=r"Threshold $2\epsilon^2$"
    )

    plt.xlabel(r"$D(p_1||p_0)$", fontsize=14)

    plt.ylabel("Empirical CDF", fontsize=14)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.7
    )

    plt.legend(fontsize=11)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig4b_CDF_D_p1_p0.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def myf_plot_Fig5(
    sys_param,
    epsilon_cand,
    MI_D01_result,
    MI_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
):
    """
    Plot Fig. 5.

    Fig. 5(a):
        MI versus epsilon

    Fig. 5(b):
        False alarm and miss detection probabilities versus epsilon
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)


    # =====================================================
    # Fig. 5(a): MI versus epsilon
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        epsilon_cand,
        MI_D01_result,
        color="blue",
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )

    plt.plot(
        epsilon_cand,
        MI_D10_result,
        color="red",
        marker="^",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )

    plt.xlabel(r"$\epsilon$", fontsize=14)

    plt.ylabel(r"Mutual information $I$ (bit)", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.legend(fontsize=11)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig5a_MI_vs_epsilon.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # Fig. 5(b): Detection probabilities
    # =====================================================

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()


    # False alarm probabilities: left y-axis
    ax1.plot(
        epsilon_cand,
        P_FA_D01_result,
        color="lime",
        marker="+",
        markersize=10,
        linewidth=2.5,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_1|\mathcal{H}_0)$"
    )

    ax1.plot(
        epsilon_cand,
        P_FA_D10_result,
        color="magenta",
        marker="^",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="magenta",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="-",
        label=r"$P_{(p_1||p_0)}(D_1|\mathcal{H}_0)$"
    )


    # Miss detection probabilities: right y-axis
    ax2.plot(
        epsilon_cand,
        P_MD_D01_result,
        color="red",
        marker="+",
        markersize=10,
        linewidth=2.5,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_0|\mathcal{H}_1)$"
    )

    ax2.plot(
        epsilon_cand,
        P_MD_D10_result,
        color="blue",
        marker="^",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="-",
        label=r"$P_{(p_1||p_0)}(D_0|\mathcal{H}_1)$"
    )

    ax1.set_xlabel(r"$\epsilon$", fontsize=14)

    ax1.set_ylabel(r"False alarm probability $P(D_1|\mathcal{H}_0)$", fontsize=14)

    ax2.set_ylabel(r"Miss detection probability $P(D_0|\mathcal{H}_1)$", fontsize=14)

    ax1.grid(True, linestyle="--", alpha=0.7)

    lines_1, labels_1 = ax1.get_legend_handles_labels()

    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
        fontsize=10,
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig5b_detection_probability_vs_epsilon.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def myf_plot_Fig6a(
    sys_param,
    v_w_cand,
    MI_D01_result,
    MI_D10_result
):
    """
    Fig. 6(a):

        Mutual information I versus Willie CSI error v_w

    Fixed:
        epsilon = 0.01

    Curves:
        D(p0 || p1)
        D(p1 || p0)
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)


    plt.figure(figsize=(8, 6))


    plt.plot(
        v_w_cand,
        MI_D01_result,
        color="blue",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.5,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )


    plt.plot(
        v_w_cand,
        MI_D10_result,
        color="red",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.5,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )


    plt.xlabel(r"$v_w$", fontsize=16)

    plt.ylabel(r"Mutual information $I$ (bit)", fontsize=16)

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tick_params(axis="both", labelsize=13)

    plt.legend(
        fontsize=12,
        loc="best",
        frameon=True
    )


    # Scientific notation like paper: x 10^{-3}
    ax = plt.gca()

    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(-3, -3)
    )


    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig6a_MI_vs_v_w.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



def myf_plot_Fig6b(
    sys_param,
    v_w_cand,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
):
    """
    Fig. 6(b):

        False alarm probability and miss detection probability
        versus Willie CSI error v_w.

    Fixed:
        epsilon = 0.05

    Left y-axis:
        P(D1 | H0)

    Right y-axis:
        P(D0 | H1)
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)


    # =====================================================
    # Convert x-axis to paper style:
    # 0.001, 0.005, ... -> 1, 5, ...
    # =====================================================

    v_w_plot = v_w_cand * 1000


    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()


    # =====================================================
    # False alarm probabilities: left y-axis
    # =====================================================

    ax1.plot(
        v_w_plot,
        P_FA_D01_result,
        color="lime",
        marker="+",
        markersize=13,
        markeredgewidth=2.2,
        linewidth=2.2,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_1|\mathcal{H}_0)$"
    )


    ax1.plot(
        v_w_plot,
        P_FA_D10_result,
        color="magenta",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="magenta",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$P_{(p_1||p_0)}(D_1|\mathcal{H}_0)$"
    )


    # =====================================================
    # Miss detection probabilities: right y-axis
    # =====================================================

    ax2.plot(
        v_w_plot,
        P_MD_D01_result,
        color="red",
        marker="+",
        markersize=13,
        markeredgewidth=2.2,
        linewidth=2.2,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_0|\mathcal{H}_1)$"
    )


    ax2.plot(
        v_w_plot,
        P_MD_D10_result,
        color="blue",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$P_{(p_1||p_0)}(D_0|\mathcal{H}_1)$"
    )


    # =====================================================
    # Axis labels
    # =====================================================

    ax1.set_xlabel(r"$v_w \;(\times 10^{-3})$", fontsize=16)

    ax1.set_ylabel(
        r"False alarm probability $P(D_1|\mathcal{H}_0)$",
        fontsize=16
    )

    ax2.set_ylabel(
        r"Miss detection probability $P(D_0|\mathcal{H}_1)$",
        fontsize=16
    )


    # =====================================================
    # Important:
    # Manual y-axis limits to avoid twin-axis visual overlap
    # =====================================================

    ax1.set_ylim(
        0.358,
        0.365
    )

    ax2.set_ylim(
        0.6205,
        0.6270
    )


    # =====================================================
    # X-axis ticks like paper
    # =====================================================

    ax1.set_xticks(v_w_plot)

    ax1.set_xticklabels(
        [str(int(x)) for x in v_w_plot]
    )


    # =====================================================
    # Grid and ticks
    # =====================================================

    ax1.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.7
    )

    ax1.tick_params(
        axis="both",
        labelsize=13
    )

    ax2.tick_params(
        axis="y",
        labelsize=13
    )


    # =====================================================
    # Combined legend
    # =====================================================

    lines_1, labels_1 = ax1.get_legend_handles_labels()

    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        fontsize=10,
        loc="upper left",
        frameon=True
    )


    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig6b_detection_probability_vs_v_w.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def myf_plot_Fig7(
    sys_param,
    N_cand,
    MI_D01_result,
    MI_D10_result
):
    """
    Fig. 7:

        Mutual information I versus number of antennas N

    Fixed:
        v_w = 0.005
        epsilon = 0.05

    Curves:
        D(p0 || p1)
        D(p1 || p0)
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)


    plt.figure(figsize=(8, 6))


    plt.plot(
        N_cand,
        MI_D01_result,
        color="blue",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.5,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )


    plt.plot(
        N_cand,
        MI_D10_result,
        color="red",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.5,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )


    plt.xlabel(r"Number of antennas $N$", fontsize=16)

    plt.ylabel(r"Mutual information $I$ (bit)", fontsize=16)

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.7
    )

    plt.tick_params(
        axis="both",
        labelsize=13
    )

    plt.legend(
        fontsize=12,
        loc="best",
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig7_MI_vs_N.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def myf_plot_Fig8(
    sys_param,
    beta_rate_cand,
    MI_D01_result,
    MI_D10_result
):
    """
    Fig. 8:
        Mutual information I versus covert rate threshold beta

    Fixed:
        v_w = 0.001
        epsilon = 0.05
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    plt.figure(figsize=(8, 6))

    plt.plot(
        beta_rate_cand,
        MI_D01_result,
        color="blue",
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )

    plt.plot(
        beta_rate_cand,
        MI_D10_result,
        color="red",
        marker="^",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.0,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )

    plt.xlabel(r"Covert rate threshold $\beta$", fontsize=16)

    plt.ylabel(r"Mutual information $I$ (bit)", fontsize=16)

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.7
    )

    plt.tick_params(axis="both", labelsize=13)

    plt.legend(
        fontsize=12,
        loc="best",
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig8_MI_vs_beta.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def myf_plot_Fig9_CDF(
    sys_param,
    D01_nonrobust,
    D10_nonrobust,
    D01_robust,
    D10_robust
):

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    KL_threshold = sys_param["KL_threshold"]


    # =====================================================
    # Fig. 9(a): D(p0 || p1)
    # =====================================================

    x_D01_nonrobust, cdf_D01_nonrobust = myf_empirical_CDF(D01_nonrobust)

    x_D01_robust, cdf_D01_robust = myf_empirical_CDF(D01_robust)


    plt.figure(figsize=(8, 6))

    plt.plot(
        x_D01_nonrobust,
        cdf_D01_nonrobust,
        color="blue",
        linewidth=2.5,
        linestyle="--",
        label="Non-robust design"
    )

    plt.plot(
        x_D01_robust,
        cdf_D01_robust,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label="Robust design"
    )

    plt.axvline(
        KL_threshold,
        color="black",
        linewidth=2.5,
        linestyle=":",
        label=r"Threshold $2\epsilon^2$"
    )

    plt.xlabel(r"$D(p_0||p_1)$", fontsize=16)

    plt.ylabel("Empirical CDF", fontsize=16)

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.legend(fontsize=12, loc="best", frameon=True)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig9a_CDF_D_p0_p1.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # Fig. 9(b): D(p1 || p0)
    # =====================================================

    x_D10_nonrobust, cdf_D10_nonrobust = myf_empirical_CDF(D10_nonrobust)

    x_D10_robust, cdf_D10_robust = myf_empirical_CDF(D10_robust)


    plt.figure(figsize=(8, 6))

    plt.plot(
        x_D10_nonrobust,
        cdf_D10_nonrobust,
        color="blue",
        linewidth=2.5,
        linestyle="--",
        label="Non-robust design"
    )

    plt.plot(
        x_D10_robust,
        cdf_D10_robust,
        color="red",
        linewidth=2.5,
        linestyle="-",
        label="Robust design"
    )

    plt.axvline(
        KL_threshold,
        color="black",
        linewidth=2.5,
        linestyle=":",
        label=r"Threshold $2\epsilon^2$"
    )

    plt.xlabel(r"$D(p_1||p_0)$", fontsize=16)

    plt.ylabel("Empirical CDF", fontsize=16)

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.legend(fontsize=12, loc="best", frameon=True)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig9b_CDF_D_p1_p0.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
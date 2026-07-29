import numpy as np

import matplotlib.pyplot as plt

import os

from fCalculations import *


'''
fPlot.py
Refat Khan

Plot functions for rate maximization imperfect-WCSI figures
Fig. 9 to Fig. 13
'''


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


def myf_plot_Fig10(
    sys_param,
    epsilon_cand,
    R_B_D01_result,
    R_B_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
):
    """
    Fig. 10:

    Fig. 10(a):
        Covert rate R_B versus epsilon

    Fig. 10(b):
        Detection probabilities versus epsilon

    Fixed:
        v_w = 0.001
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)


    # =====================================================
    # Fig. 10(a): R_B versus epsilon
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        epsilon_cand,
        R_B_D01_result,
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
        epsilon_cand,
        R_B_D10_result,
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

    plt.xlabel(r"$\epsilon$", fontsize=16)

    plt.ylabel(r"$R_B$ (bits/sec/Hz)", fontsize=16)

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
        save_folder + "/Fig10a_R_B_vs_epsilon.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # Fig. 10(b): Detection probabilities versus epsilon
    # =====================================================

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()


    # =====================================================
    # False alarm probability: left y-axis
    # =====================================================

    ax1.plot(
        epsilon_cand,
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
        epsilon_cand,
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
    # Miss detection probability: right y-axis
    # =====================================================

    ax2.plot(
        epsilon_cand,
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
        epsilon_cand,
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

    ax1.set_xlabel(r"$\epsilon$", fontsize=16)

    ax1.set_ylabel(
        r"False alarm probability $P(D_1|\mathcal{H}_0)$",
        fontsize=16
    )

    ax2.set_ylabel(
        r"Miss detection probability $P(D_0|\mathcal{H}_1)$",
        fontsize=16
    )


    # =====================================================
    # Manual y-axis limits similar to paper
    # You may adjust after seeing your result.
    # =====================================================

    ax1.set_ylim(
        0.25,
        0.40
    )

    ax2.set_ylim(
        0.40,
        0.65
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
        loc="best",
        frameon=True
    )


    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig10b_detection_probability_vs_epsilon.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt


def myf_plot_Fig11(
    sys_param,
    v_w_cand,
    R_B_D01_result,
    R_B_D10_result,
    P_FA_D01_result,
    P_FA_D10_result,
    P_MD_D01_result,
    P_MD_D10_result
):
    """
    Fig. 11

    (a) Covert rate R_B versus CSI errors v_w
    (b) Detection error probabilities versus CSI errors v_w

    Fixed:
        epsilon = 0.20
        N = 5
        P_total = 10 dBm
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    x_plot = 1e3 * v_w_cand


    # =====================================================
    # Fig. 11(a): R_B versus v_w
    # =====================================================

    plt.figure(figsize=(8, 6))

    plt.plot(
        x_plot,
        R_B_D01_result,
        color="blue",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )

    plt.plot(
        x_plot,
        R_B_D10_result,
        color="red",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )

    plt.xlabel(r"$v_w$ $(\times 10^{-3})$", fontsize=16)

    plt.ylabel(r"$R_B$ (bits/sec/Hz)", fontsize=16)

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tick_params(axis="both", labelsize=13)

    plt.legend(fontsize=12, loc="best", frameon=True)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig11a_R_B_vs_v_w.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =====================================================
    # Fig. 11(b): Detection probabilities versus v_w
    # =====================================================

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()


    # ---------------- Left axis: False alarm ----------------
    ax1.plot(
        x_plot,
        P_FA_D01_result,
        color="lime",
        marker="+",
        markersize=13,
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_1|\mathcal{H}_0)$"
    )

    ax1.plot(
        x_plot,
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


    # ---------------- Right axis: Miss detection ----------------
    ax2.plot(
        x_plot,
        P_MD_D01_result,
        color="red",
        marker="+",
        markersize=13,
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-.",
        label=r"$P_{(p_0||p_1)}(D_0|\mathcal{H}_1)$"
    )

    ax2.plot(
        x_plot,
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


    ax1.set_xlabel(r"$v_w$ $(\times 10^{-3})$", fontsize=16)

    ax1.set_ylabel(
        r"False alarm probability $P(D_1|\mathcal{H}_0)$",
        fontsize=16
    )

    ax2.set_ylabel(
        r"Miss detection probability $P(D_0|\mathcal{H}_1)$",
        fontsize=16
    )

    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    ax1.tick_params(axis="both", labelsize=13)

    ax2.tick_params(axis="y", labelsize=13)


    # You may tune these if needed after plotting
    ax1.set_ylim(0.30, 0.40)
    ax2.set_ylim(0.50, 0.62)


    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        fontsize=10,
        loc="best",
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig11b_detection_probability_vs_v_w.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt


def myf_plot_Fig12(
    sys_param,
    N_cand,
    R_B_D01_result,
    R_B_D10_result
):
    """
    Fig. 12:
    Covert rates R_B versus number of antennas N
    with CSI errors v_w = 0.001, epsilon = 0.20.
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    plt.figure(figsize=(8, 6))

    plt.plot(
        N_cand,
        R_B_D01_result,
        color="blue",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )

    plt.plot(
        N_cand,
        R_B_D10_result,
        color="red",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )

    plt.xlabel(r"Number of antennas $N$", fontsize=16)

    plt.ylabel(r"$R_B$(bits/sec/Hz)", fontsize=16)

    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tick_params(axis="both", labelsize=13)

    plt.legend(fontsize=12, loc="best", frameon=True)

    plt.tight_layout()

    plt.savefig(
        save_folder + "/Fig12_R_B_vs_N.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def myf_plot_Fig13(
    sys_param,
    gamma_cand,
    R_B_D01_result,
    R_B_D10_result
):
    """
    Fig. 13:
    Covert rates R_B versus mutual information threshold gamma
    with CSI errors v_w = 0.001, epsilon = 0.20.
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    plt.figure(figsize=(8, 6))

    plt.plot(
        gamma_cand,
        R_B_D01_result,
        color="blue",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="-",
        label=r"$D(p_0||p_1)$"
    )

    plt.plot(
        gamma_cand,
        R_B_D10_result,
        color="red",
        marker="^",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linewidth=2.2,
        linestyle="--",
        label=r"$D(p_1||p_0)$"
    )

    plt.xlabel(r"Mutual information threshold $\gamma$", fontsize=16)

    plt.ylabel(r"$R_B$(bits/sec/Hz)", fontsize=16)

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
        save_folder + "/Fig13_R_B_vs_gamma.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
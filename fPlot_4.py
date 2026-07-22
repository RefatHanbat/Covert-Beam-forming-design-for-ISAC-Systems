import numpy as np

import matplotlib.pyplot as plt

import os

def myf_get_x_label(x_axis_name):
    """
    Return x-axis label based on x_axis_name.
    """

    if x_axis_name == "P_total_dBm_cand":

        x_label = r"$P_{\mathrm{total}}$ (dBm)"

    elif x_axis_name == "N_cand":

        x_label = r"Number of antennas $N$"

    else:

        x_label = x_axis_name

    return x_label

def myf_plot_MI_R_and_R_B(sys_param, x_axis_cand, MI_R_result, R_B_result, x_axis_name):
    """
    Plot Fig. 2 / Fig. 3 style curves using current row-based coding style.

    Row meaning:
        MI_R_result[0, :] = Covert beamformer I_R
        R_B_result[1, :]  = Covert beamformer R_B
        MI_R_result[2, :] = ZF beamformer I_R
        R_B_result[3, :]  = ZF beamformer R_B
    """

    save_folder = "figures"

    if not os.path.exists(save_folder):

        os.makedirs(save_folder)

    x_label = myf_get_x_label(x_axis_name)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax2 = ax1.twinx()


    # =====================================================
    # Row 0: Covert beamformer I_R
    # =====================================================
    ax1.plot(
        x_axis_cand,
        MI_R_result[0, :],
        color="lime",
        marker="*",
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linewidth=2.5,
        linestyle="-",
        label=r"Covert beamformers $I_R$"
    )


    # =====================================================
    # Row 1: Covert beamformer R_B
    # =====================================================
    ax2.plot(
        x_axis_cand,
        R_B_result[1, :],
        color="red",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="red",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="-",
        label=r"Covert beamformers $R_B$"
    )


    # =====================================================
    # Row 2: ZF beamformer I_R
    # =====================================================
    ax1.plot(
        x_axis_cand,
        MI_R_result[2, :],
        color="blue",
        marker="*",
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linewidth=2.5,
        linestyle=":",
        label=r"ZF beamformers $I_R$"
    )


    # =====================================================
    # Row 3: ZF beamformer R_B
    # =====================================================
    ax2.plot(
        x_axis_cand,
        R_B_result[3, :],
        color="magenta",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="magenta",
        markeredgewidth=1.8,
        linewidth=2.5,
        linestyle="-",
        label=r"ZF beamformers $R_B$"
    )


    # =====================================================
    # Axis labels
    # =====================================================
    ax1.set_xlabel(x_label, fontsize=14)

    ax1.set_ylabel(r"Mutual information $I$ (bit)", fontsize=14)

    ax2.set_ylabel(r"$R_B$ (bits/sec/Hz)", fontsize=14)


    # =====================================================
    # Grid and ticks
    # =====================================================
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    ax1.tick_params(axis="both", labelsize=12)

    ax2.tick_params(axis="y", labelsize=12)

    ax2.ticklabel_format(useOffset=False, style="plain", axis="y")


    # =====================================================
    # Combined legend
    # =====================================================
    lines_1, labels_1 = ax1.get_legend_handles_labels()

    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()

    plt.savefig(
        save_folder + "/MI_R_and_R_B_vs_" + x_axis_name + ".png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
import matplotlib.pyplot as plt
import numpy as np

Omega_0 = 2 * 1e4
H = 2 * 1e9  # GeV
lambda_values = [1e-10, 1e-12, 1e-14]

# x-values
fa = np.logspace(13, 18, 400)

fig, axes = plt.subplots(
    1, 3, figsize=(18, 6),
    sharex=True, sharey=True
    #gridspec_kw={"wspace": 0.1}
)

for ax, lambda_rho in zip(axes, lambda_values):

    # ------------------------------------------------------------
    # Boundaries
    # ------------------------------------------------------------
    y1 = 1e9 * np.sqrt(lambda_rho) * np.ones_like(fa)
    y2 = 1e5 * np.sqrt(lambda_rho) * np.ones_like(fa)

    y_constraint = 1e36 / fa**2

    # Isocurvature bounds
    lower_limit_isocurvature = (
        lambda_rho * 1e12/(12*(2*np.pi*4.6)**2)*(fa*1e-12)**(7/6)
        - lambda_rho*fa**2/(12*H**2*Omega_0**2)
    )
    lower_limit_isocurvature_2 = (
        lambda_rho * 1e12/(12*(2*np.pi*4.6)**2)*(fa*1e-12)**(7/6)
        - lambda_rho*fa**2/(12*H**2)
    )

    # ------------------------------------------------------------
    # Build un-clipped bands
    # ------------------------------------------------------------
    red_lower    = y2
    red_upper    = y1

    orange_lower = lower_limit_isocurvature
    orange_upper = np.minimum(y2, y1)

    green_lower  = lower_limit_isocurvature
    green_upper  = lower_limit_isocurvature_2

    # ------------------------------------------------------------
    # Clipping helper
    # ------------------------------------------------------------
    def clip_band(lower, upper):
        lower_c = np.maximum(lower, lower_limit_isocurvature_2)
        upper_c = np.minimum(upper, y_constraint)
        mask = upper_c > lower_c
        return lower_c, upper_c, mask

    # Main regions
    red_low_c, red_up_c, red_mask     = clip_band(red_lower, red_upper)
    orange_low_c, orange_up_c, o_mask = clip_band(orange_lower, orange_upper)
    green_low_c, green_up_c, g_mask   = clip_band(green_lower, green_upper)

    # Extra red region below y2
    extra_low_c, extra_up_c, extra_mask = clip_band(lower_limit_isocurvature_2, y2)
    extra_mask &= ~o_mask & ~g_mask

    # ------------------------------------------------------------
    # Shaded regions  (EDGES DISABLED)
    # ------------------------------------------------------------
    ax.fill_between(
        fa, red_low_c, red_up_c,
        where=red_mask,
        color="#DD0426", alpha=0.25,
        edgecolor="none", linewidth=0
    )

    ax.fill_between(
        fa, orange_low_c, orange_up_c,
        where=o_mask,
        color="#32936F", alpha=0.25,
        edgecolor="none", linewidth=0
    )

    ax.fill_between(
        fa, green_low_c, green_up_c,
        where=g_mask,
        color="#32936F", alpha=0.25,
        edgecolor="none", linewidth=0
    )

    ax.fill_between(
        fa, extra_low_c, extra_up_c,
        where=extra_mask,
        color="#DD0426", alpha=0.25,
        edgecolor="none", linewidth=0
    )

    # ------------------------------------------------------------
    # Isocurvature curves
    # ------------------------------------------------------------
    ax.plot(
        fa, lower_limit_isocurvature, "-", color="#32936F", lw=5,
        label="Isocurvature bound"
        if lambda_rho==lambda_values[0] else None
    )
    ax.plot(
        fa, lower_limit_isocurvature_2, "--", color="#DD0426", lw=3,
        label="Isocurvature bound minimally coupled"
        if lambda_rho==lambda_values[0] else None
    )

    # ------------------------------------------------------------
    # Horizontal dashed y1 plateau
    # ------------------------------------------------------------
    plateau_mask = red_mask & (np.abs(red_up_c - y1) < 1e-12)

    ax.plot(
        fa[plateau_mask], y1[plateau_mask],
        "--k", lw=2,
        label="Upper bound minimally coupled" if lambda_rho==lambda_values[0] else None
    )

    # ------------------------------------------------------------
    # Solid black: lower boundary of red region
    # ------------------------------------------------------------
    total_low = np.where(extra_mask, extra_low_c, red_low_c)
    total_mask = o_mask

    ax.plot(
        fa[total_mask],
        total_low[total_mask],
        "-k", lw=4,
        label="Upper bound non-minimally coupled" if lambda_rho==lambda_values[0] else None
    )

    # ------------------------------------------------------------
    # Slanted dashed y_constraint segment
    # ------------------------------------------------------------
    constraint_mask = (
        red_mask &
        (np.abs(red_up_c - y_constraint) < 1e-12)
    )

    ax.plot(
        fa[constraint_mask],
        y_constraint[constraint_mask],
        "--k", lw=2.2
    )

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"$\lambda_\rho = 10^{%d}$" % np.log10(lambda_rho), fontsize=22)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.tick_params(axis="both", labelsize=18)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------
fig.text(0.5, 0.02, r"$f_a$ [GeV]", ha="center", fontsize=22)
fig.text(0.04, 0.5, r"$\xi_\rho$",  va="center", rotation="vertical", fontsize=22)

handles, labels = axes[0].get_legend_handles_labels()

desired_order = [
    "Isocurvature bound",
    "Upper bound non-minimally coupled",
    "Isocurvature bound minimally coupled",
    "Upper bound minimally coupled",
]

# Reorder handles according to desired_order
ordered_handles = []
ordered_labels = []

for name in desired_order:
    if name in labels:
        idx = labels.index(name)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])

axes[0].legend(
    ordered_handles,
    ordered_labels,
    loc="lower left",
    fontsize=14
)
axes[0].set_ylim(1e-7, 1e5)
ax.set_xlim(fa.min(), fa.max())
ax.margins(x=0)

plt.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.savefig("Palatini_final.png", bbox_inches="tight")
plt.show()

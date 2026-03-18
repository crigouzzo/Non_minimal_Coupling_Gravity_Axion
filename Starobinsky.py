import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------
N = 50
H = 1e13
Mp = 2 * 10**18

lambda_values = [1e-9, 1e-11, 1e-13]

# -------------------------------------------------------------------
# Effective decay constant during inflation
# -------------------------------------------------------------------
def fainf(fa, xirho, lambdarho, N):
    num = 1 + 16*N*xirho*H**2/(lambdarho*fa**2)
    den = 1 + 12*xirho**2*H**2/(lambdarho*Mp**2)
    return fa * np.sqrt(3/(4*N)) * np.sqrt(num/den)

def T(fa):
    return H * 1e6 / (2*np.pi*4.6) * (fa*1e-12)**(7/12)

# -------------------------------------------------------------------
# Solve isocurvature curve branches
# -------------------------------------------------------------------
def isocurvature_branches(fa, lambdarho):
    Tval = T(fa)
    C = 3 * fa**2 / (4*N)

    A = 16*N*H**2/(lambdarho*fa**2)
    B = 12*H**2/(lambdarho*Mp**2)

    a = Tval**2 * B
    b = -C * A
    c = Tval**2 - C

    disc = b**2 - 4*a*c

    xi_minus = np.full_like(fa, np.nan)
    xi_plus  = np.full_like(fa, np.nan)

    mask = disc >= 0
    sqrt_disc = np.sqrt(disc[mask])

    xi_minus[mask] = (-b[mask] - sqrt_disc) / (2 * a[mask])
    xi_plus[mask]  = (-b[mask] + sqrt_disc) / (2 * a[mask])

    return xi_minus, xi_plus


# -------------------------------------------------------------------
# Grid for classification
# -------------------------------------------------------------------
fa = np.logspace(12, 15, 2000)
xi = np.logspace(-12, 2, 4000)
X, Y = np.meshgrid(fa, xi)

# -------------------------------------------------------------------
# Prepare 3-panel figure
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, lambdarho in zip(axes, lambda_values):

    # Evaluate mask Z > T
    Z = fainf(X, Y, lambdarho, N)
    Thresh = T(X)
    mask = Z > Thresh

    # minimal-coupling isocurvature limit
    lower_limit_isocurvature_2 = (
        lambdarho*1e12/(12*(2*np.pi*4.6)**2)*(fa*1e-12)**(7/6)
        - lambdarho*fa**2/(12*H**2)
    )
    mask2 = Y > lower_limit_isocurvature_2

    # region boundaries
    y1 = 1e5 * np.sqrt(lambdarho)
    y2 = 1e4 * np.sqrt(lambdarho)

    mask_red   = mask2 & (Y > y2) & (Y < y1)
    mask_beige = mask2 & (Y < y2)

    # shaded envelopes
    beige_min = np.full_like(fa, np.nan)
    beige_max = np.full_like(fa, np.nan)
    red_min   = np.full_like(fa, np.nan)
    red_max   = np.full_like(fa, np.nan)

    for i in range(len(fa)):
        col_xi = Y[:, i]

        vals = col_xi[mask_beige[:, i]]
        if vals.size > 0:
            beige_min[i] = vals.min()
            beige_max[i] = vals.max()

        vals = col_xi[mask_red[:, i]]
        if vals.size > 0:
            red_min[i] = vals.min()
            red_max[i] = vals.max()

    # isocurvature branches
    xi_minus, xi_plus = isocurvature_branches(fa, lambdarho)
    mask_minus = np.isfinite(xi_minus) & (xi_minus > 0)
    mask_plus  = np.isfinite(xi_plus)  & (xi_plus  > 0)

    # --- beige shading ---
    ax.fill_between(
        fa, beige_min, beige_max,
        where=np.isfinite(beige_min) & np.isfinite(beige_max),
        color="#32936F", alpha=0.25
    )

    # --- red shading ---
    ax.fill_between(
        fa, red_min, red_max,
        where=np.isfinite(red_min) & np.isfinite(red_max),
        color="#DD0426", alpha=0.25
    )

    # --- isocurvature bounds ---
    ax.plot(fa[mask_minus], xi_minus[mask_minus], "-", color="#32936F", lw=5)
    ax.plot(
        fa[mask_plus], xi_plus[mask_plus],
        "-", color="#32936F", lw=5,
        label="Isocurvature bound" if lambdarho == lambda_values[0] else None
    )

    ax.plot(
        fa, lower_limit_isocurvature_2, "--", color="#DD0426", lw=3,
        label="Isocurvature bound minimally coupled" if lambdarho == lambda_values[0] else None
    )

    # --- minimal coupling (horizontal y1) ---
    mask_y1 = lower_limit_isocurvature_2 < y1
    ax.plot(
        fa[mask_y1],
        np.full_like(fa[mask_y1], y1),
        "--k", lw=2,
        label="Upper bound minimally coupled" if lambdarho == lambda_values[0] else None
    )

    # --- non-minimal coupling boundary (horizontal y2) ---
    mask_y2 = (y2 > xi_minus) & np.isfinite(xi_minus)
    ax.plot(
        fa[mask_y2],
        np.full_like(fa[mask_y2], y2),
        "-k", lw=4,
        label="Upper bound non-minimally coupled" if lambdarho == lambda_values[0] else None
    )

    # formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e1)
    ax.set_xlim(1e12, 1e15)

    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.set_title(r"$\lambda_\rho = 10^{%d}$" % np.log10(lambdarho), fontsize=22)
    ax.tick_params(axis="both", labelsize=20)

# -------------------------------------------------------------------
# Custom x ticks (hide leftmost tick on last two subplots)
# -------------------------------------------------------------------
ticks = [1e12, 1e13, 1e14, 1e15]
ticklabels_full = [r"$10^{12}$", r"$10^{13}$", r"$10^{14}$", r"$10^{15}$"]
ticklabels_trim = ["", r"$10^{13}$", r"$10^{14}$", r"$10^{15}$"]

axes[0].set_xticks(ticks)
axes[0].set_xticklabels(ticklabels_full)

axes[1].set_xticks(ticks)
axes[1].set_xticklabels(ticklabels_trim)

axes[2].set_xticks(ticks)
axes[2].set_xticklabels(ticklabels_trim)

# -------------------------------------------------------------------
# Labels and spacing
# -------------------------------------------------------------------
fig.text(0.5, 0.02, r"$f_a$ [GeV]", ha="center", fontsize=22)
fig.text(0.03, 0.5, r"$\xi_\rho$", va="center", rotation="vertical", fontsize=22)

# ------------------------------------------------------------
# Legend with custom order
# ------------------------------------------------------------
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


fig.tight_layout(rect=[0.06, 0.06, 1, 1])
plt.subplots_adjust(wspace=0.05)

plt.savefig("Starobinsky_final", bbox_inches='tight')
plt.show()

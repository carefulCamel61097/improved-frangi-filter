"""Generate all figures for the Improved Frangi Filter project.

Run this script to reproduce every figure used in the README:

    python generate_figures.py

Figures are saved to the figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import maximum_filter, minimum_filter
from frangi_filter import hessian_2d, eigenvalues_2d, frangi_2d

SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "figures"
IMG_PATH = SCRIPT_DIR / "dsa_sample.jpg"

SIGMA_WIDE = (2, 20)
BETA1, BETA2 = 0.5, 15.0


def load_image() -> np.ndarray:
    image = plt.imread(IMG_PATH)
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    return image.astype(np.float64)


def save(fig, name: str):
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figures/{name}")


# ── Figure 1: The blur problem ──────────────────────────────────────────────

def fig1_blur_problem(image: np.ndarray):
    """Standard Frangi at narrow vs wide sigma range — shows the blur."""
    print("Figure 1: Blur problem...")
    result_narrow = frangi_2d(image, sigma_range=(2, 7), bright_on_dark=False)
    result_wide = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original DSA Image")
    axes[1].imshow(result_narrow, cmap="hot")
    axes[1].set_title("Frangi — narrow range (σ = 2–7)")
    axes[2].imshow(result_wide, cmap="hot")
    axes[2].set_title("Frangi — wide range (σ = 2–20)\n(note the blur)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("The Blur Problem: Narrow vs Wide Scale Range", fontsize=14)
    fig.tight_layout()
    save(fig, "01_blur_problem.png")


# ── Figure 2: Per-scale intermediates ────────────────────────────────────────

def fig2_intermediates(image: np.ndarray):
    """Eigenvalues, Frangi response, S3, and total at three scales."""
    print("Figure 2: Per-scale intermediates...")
    sigmas = [2, 7, 20]

    fig, axes = plt.subplots(len(sigmas), 5, figsize=(20, 4 * len(sigmas)))

    for row, sigma in enumerate(sigmas):
        Hxx, Hyy, Hxy = hessian_2d(image, sigma)
        lambda1, lambda2 = eigenvalues_2d(Hxx, Hyy, Hxy)

        mask = lambda2 > 0
        lambda2_safe = np.where(lambda2 == 0, 1e-10, lambda2)
        S1 = (lambda1 / lambda2_safe) ** 2
        S2 = lambda1**2 + lambda2**2
        frangi = np.exp(-S1 / BETA1) * (1 - np.exp(-S2 / BETA2))
        frangi = np.where(mask, frangi, 0)

        kernel_px = 6 * sigma + 1
        I_max = maximum_filter(image, size=kernel_px)
        I_min = minimum_filter(image, size=kernel_px)
        denom = I_max - I_min
        safe_denom = np.where(denom > 0, denom, 1.0)
        S3 = np.where(denom > 0, (I_max - image) / safe_denom, 0.0)

        total = frangi * S3

        panels = [
            (lambda1, f"λ1 (σ={sigma})", "signed"),
            (lambda2, f"λ2 (σ={sigma})", "signed"),
            (frangi, f"Frangi response (σ={sigma})", "hot"),
            (S3, f"S3 local normalization (σ={sigma})", "gray"),
            (total, f"Total: Frangi × S3 (σ={sigma})", "hot"),
        ]

        for col, (data, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            if cmap == "signed":
                vmax = np.percentile(np.abs(data), 99)
                ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            elif cmap == "gray":
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(data, cmap="hot", vmin=0, vmax=np.percentile(data, 99))
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("Frangi Filter Intermediates at Different Scales", fontsize=14)
    fig.tight_layout()
    save(fig, "02_intermediates.png")


# ── Figure 3: Local normalization improvement ────────────────────────────────

def fig3_local_normalization(image: np.ndarray):
    """Side-by-side: standard wide vs wide + local normalization."""
    print("Figure 3: Local normalization improvement...")
    result_wide = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False)
    result_norm = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                            local_normalization=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original DSA Image")
    axes[1].imshow(result_wide, cmap="hot")
    axes[1].set_title("Standard Frangi (σ = 2–20)")
    axes[2].imshow(result_norm, cmap="hot")
    axes[2].set_title("+ Local Normalization (S3)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Improvement 1: Local Intensity Normalization", fontsize=14)
    fig.tight_layout()
    save(fig, "03_local_normalization.png")


# ── Figure 4: Winning sigma map ──────────────────────────────────────────────

def fig4_winning_sigma(image: np.ndarray):
    """Color map of which sigma wins at each pixel."""
    print("Figure 4: Winning sigma map...")
    sigmas = list(range(SIGMA_WIDE[0], SIGMA_WIDE[1] + 1))
    all_responses = np.zeros((len(sigmas), *image.shape))

    for i, sigma in enumerate(sigmas):
        Hxx, Hyy, Hxy = hessian_2d(image, sigma)
        lambda1, lambda2 = eigenvalues_2d(Hxx, Hyy, Hxy)

        mask = lambda2 > 0
        lambda2_safe = np.where(lambda2 == 0, 1e-10, lambda2)
        S1 = (lambda1 / lambda2_safe) ** 2
        S2 = lambda1**2 + lambda2**2
        response = np.exp(-S1 / BETA1) * (1 - np.exp(-S2 / BETA2))
        response = np.where(mask, response, 0)

        kernel_size = 6 * sigma + 1
        I_max = maximum_filter(image, size=kernel_size)
        I_min = minimum_filter(image, size=kernel_size)
        denom = I_max - I_min
        safe_denom = np.where(denom > 0, denom, 1.0)
        S3 = np.where(denom > 0, (I_max - image) / safe_denom, 0.0)
        all_responses[i] = response * S3

    winning_idx = np.argmax(all_responses, axis=0)
    winning_sigma = np.array(sigmas)[winning_idx]
    max_response = np.max(all_responses, axis=0)
    has_response = max_response > max_response.max() * 0.01
    sigma_display = np.where(has_response, winning_sigma.astype(float), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original DSA Image")
    axes[0].axis("off")

    im = axes[1].imshow(sigma_display, cmap="turbo",
                         vmin=sigmas[0], vmax=sigmas[-1])
    axes[1].set_title("Winning σ per pixel")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="σ value")

    axes[2].imshow(max_response, cmap="gray", vmin=0,
                    vmax=np.percentile(max_response, 99))
    overlay = axes[2].imshow(sigma_display, cmap="turbo", alpha=0.6,
                              vmin=sigmas[0], vmax=sigmas[-1])
    axes[2].set_title("Winning σ overlaid on vesselness")
    axes[2].axis("off")
    fig.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04, label="σ value")

    fig.suptitle("Winning Sigma Map (σ = 2–20, with local normalization)", fontsize=14)
    fig.tight_layout()
    save(fig, "04_winning_sigma.png")


# ── Figure 5: Dominant scale variants ────────────────────────────────────────

def fig5_dominant_scale_variants(image: np.ndarray):
    """Compare the four dominant scale approaches."""
    print("Figure 5: Dominant scale variants...")
    result_norm = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                            local_normalization=True)

    variants = {
        "Post-norm\nunweighted": dict(dominant_scale=True),
        "Pre-norm\nunweighted": dict(dominant_scale=True, dominant_scale_pre_norm=True),
        "Post-norm\nweighted": dict(dominant_scale=True, dominant_scale_weighted=True),
        "Pre-norm\nweighted": dict(dominant_scale=True, dominant_scale_pre_norm=True,
                                    dominant_scale_weighted=True),
    }

    results = {}
    for label, kwargs in variants.items():
        results[label] = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                                    local_normalization=True, **kwargs)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original DSA Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result_norm, cmap="hot")
    axes[0, 1].set_title("Local normalization only\n(baseline)")
    axes[0, 1].axis("off")

    axes[0, 2].set_axis_off()

    for i, (label, result) in enumerate(results.items()):
        row, col = divmod(i, 3)
        if row == 0:
            col += 2  # won't happen with 4 items starting at bottom row
        ax = axes[1, i] if i < 3 else axes[0, 2]
        # Place all 4 in bottom row + top-right
        pass

    # Manually place for clean layout
    labels = list(results.keys())
    axes[1, 0].imshow(results[labels[0]], cmap="hot")
    axes[1, 0].set_title(labels[0])
    axes[1, 0].axis("off")

    axes[1, 1].imshow(results[labels[1]], cmap="hot")
    axes[1, 1].set_title(labels[1])
    axes[1, 1].axis("off")

    axes[1, 2].imshow(results[labels[2]], cmap="hot")
    axes[1, 2].set_title(labels[2])
    axes[1, 2].axis("off")

    axes[0, 2].imshow(results[labels[3]], cmap="hot")
    axes[0, 2].set_title(labels[3])
    axes[0, 2].axis("off")

    fig.suptitle("Improvement 2: Dominant Scale Filter Variants (threshold=50%)", fontsize=14)
    fig.tight_layout()
    save(fig, "05_dominant_scale_variants.png")


# ── Figure 6: Threshold comparison ───────────────────────────────────────────

def fig6_threshold_comparison(image: np.ndarray):
    """Fine-grained threshold sweep for weighted post-norm dominant scale."""
    print("Figure 6: Threshold comparison...")
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = {}
    for t in thresholds:
        label = f"{int(t * 100)}%"
        results[label] = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                                    local_normalization=True, dominant_scale=True,
                                    dominant_scale_weighted=True,
                                    dominant_scale_threshold=t)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (label, result) in enumerate(results.items()):
        row, col = divmod(i, 3)
        axes[row, col].imshow(result, cmap="hot")
        axes[row, col].set_title(f"Threshold = {label}")
        axes[row, col].axis("off")

    fig.suptitle("Dominant Scale: Threshold Fine-Tuning (post-norm, weighted)", fontsize=14)
    fig.tight_layout()
    save(fig, "06_threshold_comparison.png")


# ── Figure 7: Final comparison ───────────────────────────────────────────────

def fig7_final_comparison(image: np.ndarray):
    """The full story: original → standard → + normalization → + dominant scale."""
    print("Figure 7: Final comparison...")
    result_standard = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False)
    result_norm = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                            local_normalization=True)
    result_full = frangi_2d(image, sigma_range=SIGMA_WIDE, bright_on_dark=False,
                            local_normalization=True, dominant_scale=True,
                            dominant_scale_weighted=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original DSA Image")
    axes[1].imshow(result_standard, cmap="hot")
    axes[1].set_title("Standard Frangi\n(σ = 2–20)")
    axes[2].imshow(result_norm, cmap="hot")
    axes[2].set_title("+ Local Normalization")
    axes[3].imshow(result_full, cmap="hot")
    axes[3].set_title("+ Weighted Dominant Scale")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Improved Frangi Filter: Full Pipeline", fontsize=14)
    fig.tight_layout()
    save(fig, "07_final_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    image = load_image()

    fig1_blur_problem(image)
    fig2_intermediates(image)
    fig3_local_normalization(image)
    fig4_winning_sigma(image)
    fig5_dominant_scale_variants(image)
    fig6_threshold_comparison(image)
    fig7_final_comparison(image)

    print("\nAll figures generated.")

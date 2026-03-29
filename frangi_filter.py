import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter


def hessian_2d(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 2D Hessian matrix elements of an image at a given scale.

    Uses Gaussian second-order derivatives with scale-space normalization
    (multiplying by sigma^2) as described by Frangi et al.
    """
    smoothed = gaussian_filter(image.astype(np.float64), sigma)

    # Second-order partial derivatives via finite differences on the smoothed image
    Hxx = np.gradient(np.gradient(smoothed, axis=1), axis=1) * sigma**2
    Hyy = np.gradient(np.gradient(smoothed, axis=0), axis=0) * sigma**2
    Hxy = np.gradient(np.gradient(smoothed, axis=0), axis=1) * sigma**2

    return Hxx, Hyy, Hxy


def eigenvalues_2d(Hxx: np.ndarray, Hyy: np.ndarray, Hxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute sorted eigenvalues of the 2D Hessian at every pixel.

    Returns (lambda1, lambda2) where |lambda1| <= |lambda2|.
    """
    trace = Hxx + Hyy
    det = Hxx * Hyy - Hxy**2
    discriminant = np.sqrt(np.maximum((trace / 2)**2 - det, 0))

    eig1 = trace / 2 + discriminant
    eig2 = trace / 2 - discriminant

    # Sort so |lambda1| <= |lambda2|
    abs1, abs2 = np.abs(eig1), np.abs(eig2)
    lambda1 = np.where(abs1 <= abs2, eig1, eig2)
    lambda2 = np.where(abs1 <= abs2, eig2, eig1)

    return lambda1, lambda2


def dominant_scale_filter(responses: np.ndarray,
                          winning_sigmas: np.ndarray,
                          sigmas: list[int],
                          outer_size: int,
                          inner_size: int,
                          threshold: float = 0.2,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Suppress scale outliers by enforcing local scale consensus.

    For each position, looks at a large (outer) window to determine which
    sigmas dominate, then restricts the response in the center (inner) region
    to only those dominant scales.

    Parameters
    ----------
    responses : 3D array (n_sigmas, H, W)
        Frangi response at each scale.
    winning_sigmas : 2D array (H, W)
        The sigma index that gave the max response at each pixel.
    sigmas : list of int
        The sigma values used.
    outer_size : int
        Size of the analysis window.
    inner_size : int
        Size of the center region where the scale restriction is applied.
    threshold : float
        Fraction (0-1) of the most frequent sigma's weight. Sigmas with
        less than this fraction are suppressed. Higher = more aggressive.
    weights : 2D array (H, W) or None
        Per-pixel weight for the sigma vote. If None, each pixel votes
        equally (unweighted count). If provided, each pixel's vote is
        weighted by this value (e.g. its response strength).

    Returns
    -------
    vesselness : 2D array
        Filtered max-across-scales response.
    """
    H, W = winning_sigmas.shape
    n_sigmas = len(sigmas)
    vesselness = np.max(responses, axis=0)

    pad = outer_size // 2

    # Pad for border handling
    winning_padded = np.pad(winning_sigmas, pad, mode="edge")
    if weights is not None:
        weights_padded = np.pad(weights, pad, mode="edge")

    # Slide the outer window, apply restriction to the inner center
    for y in range(0, H, inner_size):
        for x in range(0, W, inner_size):
            # Outer window bounds (in padded coordinates)
            oy1 = y
            oy2 = min(y + outer_size, H + 2 * pad)
            ox1 = x
            ox2 = min(x + outer_size, W + 2 * pad)

            # Compute sigma frequencies in outer window
            outer_winners = winning_padded[oy1:oy2, ox1:ox2].ravel()

            if weights is not None:
                outer_weights = weights_padded[oy1:oy2, ox1:ox2].ravel()
                counts = np.bincount(outer_winners, weights=outer_weights,
                                     minlength=n_sigmas)
            else:
                counts = np.bincount(outer_winners, minlength=n_sigmas)

            # Find dominant sigmas
            if counts.max() == 0:
                continue
            min_count = counts.max() * threshold
            allowed = np.where(counts >= min_count)[0]

            # Inner region bounds (in original coordinates)
            iy1 = y
            iy2 = min(y + inner_size, H)
            ix1 = x
            ix2 = min(x + inner_size, W)

            # Rebuild vesselness using only allowed scales
            inner_responses = responses[:, iy1:iy2, ix1:ix2]
            mask = np.zeros(n_sigmas, dtype=bool)
            mask[allowed] = True
            masked_responses = np.where(mask[:, None, None], inner_responses, 0)
            vesselness[iy1:iy2, ix1:ix2] = np.max(masked_responses, axis=0)

    return vesselness


def frangi_2d(image: np.ndarray,
              sigma_range: tuple[int, int] = (2, 7),
              beta1: float = 0.5,
              beta2: float = 15.0,
              bright_on_dark: bool = True,
              local_normalization: bool = False,
              dominant_scale: bool = False,
              dominant_scale_inner: int | None = None,
              dominant_scale_threshold: float = 0.5,
              dominant_scale_weighted: bool = False,
              dominant_scale_pre_norm: bool = False) -> np.ndarray:
    """2D Frangi vesselness filter with optional improvements.

    Parameters
    ----------
    image : 2D array
        Input grayscale image.
    sigma_range : tuple
        (min_sigma, max_sigma) inclusive range for the Gaussian scale levels.
    beta1 : float
        Correction constant for the blobness factor S1.
    beta2 : float
        Correction constant for the structureness factor S2.
    bright_on_dark : bool
        If True, detect bright vessels on dark background (lambda2 < 0).
        If False, detect dark vessels on bright background (lambda2 > 0).
    local_normalization : bool
        If True, apply the S3 local intensity normalization factor
        to suppress blur from the multiscale approach.
    dominant_scale : bool
        If True, apply dominant scale filtering to suppress scale outliers.
    dominant_scale_inner : int or None
        Inner window size for dominant scale filter. Defaults to
        6 * sigma_min + 1 if not specified.
    dominant_scale_threshold : float
        Fraction (0-1) of the most frequent sigma's weight below which
        sigmas are suppressed. Higher = more aggressive filtering.
    dominant_scale_weighted : bool
        If True, weight each pixel's sigma vote by its response strength.
    dominant_scale_pre_norm : bool
        If True, determine dominant scales from pre-normalization responses
        (raw Frangi), but still apply normalization to the final output.

    Returns
    -------
    vesselness : 2D array
        Frangi vesselness response (max across scales).
    """
    image = image.astype(np.float64)
    sigmas = list(range(sigma_range[0], sigma_range[1] + 1))
    raw_responses = np.zeros((len(sigmas), *image.shape))
    norm_responses = np.zeros((len(sigmas), *image.shape))

    for i, sigma in enumerate(sigmas):
        Hxx, Hyy, Hxy = hessian_2d(image, sigma)
        lambda1, lambda2 = eigenvalues_2d(Hxx, Hyy, Hxy)

        # Suppress based on vessel polarity
        if bright_on_dark:
            mask = lambda2 < 0
        else:
            mask = lambda2 > 0

        # S1: blobness ratio
        lambda2_safe = np.where(lambda2 == 0, 1e-10, lambda2)
        S1 = (lambda1 / lambda2_safe) ** 2

        # S2: structureness
        S2 = lambda1**2 + lambda2**2

        # Frangi response
        response = np.exp(-S1 / beta1) * (1 - np.exp(-S2 / beta2))
        response = np.where(mask, response, 0)
        raw_responses[i] = response

        # S3: local intensity normalization
        if local_normalization:
            kernel_size = 6 * sigma + 1
            I_max = maximum_filter(image, size=kernel_size)
            I_min = minimum_filter(image, size=kernel_size)
            denom = I_max - I_min
            safe_denom = np.where(denom > 0, denom, 1.0)
            if bright_on_dark:
                S3 = np.where(denom > 0, (image - I_min) / safe_denom, 0.0)
            else:
                S3 = np.where(denom > 0, (I_max - image) / safe_denom, 0.0)
            norm_responses[i] = response * S3
        else:
            norm_responses[i] = response

    # Choose which responses to use for output and for scale analysis
    output_responses = norm_responses
    if dominant_scale_pre_norm:
        analysis_responses = raw_responses
    else:
        analysis_responses = norm_responses

    if dominant_scale:
        winning_sigmas = np.argmax(analysis_responses, axis=0).astype(np.intp)
        max_response = np.max(analysis_responses, axis=0)
        outer_size = 6 * sigmas[-1] + 1
        inner_size = dominant_scale_inner or (6 * sigmas[0] + 1)
        weights = max_response if dominant_scale_weighted else None
        vesselness = dominant_scale_filter(
            output_responses, winning_sigmas, sigmas, outer_size, inner_size,
            dominant_scale_threshold, weights)
    else:
        vesselness = np.max(output_responses, axis=0)

    return vesselness

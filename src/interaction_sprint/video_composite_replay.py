"""Post-generation composite contrasts, not a model or effect-mask estimator.

Arrays must already share one aligned pixel coordinate system. Reference pixels
are used only for scoring; this module never calls a generator.
"""
import numpy as np


def composite(source, generated, alpha):
    source, generated, alpha = map(np.asarray, (source, generated, alpha))
    if source.shape != generated.shape or source.ndim < 3 or source.shape[-1] != 3:
        raise ValueError('Expected matching RGB image/video shapes')
    if alpha.shape != source.shape[:-1]:
        raise ValueError('Alpha must match pixel dimensions exactly')
    if not all(np.isfinite(x).all() for x in (source, generated, alpha)):
        raise ValueError('Nonfinite input')
    if any(x.min() < 0 or x.max() > 1 for x in (source, generated, alpha)):
        raise ValueError('Expected normalized [0,1] inputs')
    return generated * alpha[..., None] + source * (1-alpha[..., None])


def replay(source, generated, reference, alphas, regions):
    """Float-domain MAE only; no perceptual, physical, or paper decision."""
    source, generated, reference = map(np.asarray, (source, generated, reference))
    composite(source, reference, np.ones(source.shape[:-1]))  # validate reference
    results = {}
    for name, alpha in alphas.items():
        output = composite(source, generated, alpha)
        scores = {}
        for region_name, region in regions.items():
            region = np.asarray(region)
            if region.dtype != bool or region.shape != source.shape[:-1] or not region.any():
                raise ValueError('Regions must be nonempty boolean pixel masks')
            # For alpha=0, no possible generated content can reduce this error.
            fixed = region & (np.asarray(alpha) == 0)
            floor = np.abs(source-reference)[fixed].sum() / (region.sum()*3)
            scores[region_name] = dict(mae=float(np.abs(output-reference)[region].mean()),
                change_from_source=float(np.abs(output-source)[region].mean()),
                hard_support_error_floor=float(floor),
                fixed_pixel_fraction=float(fixed.sum()/region.sum()))
        results[name] = scores
    return results

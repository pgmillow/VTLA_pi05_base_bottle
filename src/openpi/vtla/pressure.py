"""Calibration for the native O10 268-channel pressure vector (no spatial guess)."""
import numpy as np


def calibrate_o10(frames, valid, *, min_valid_frames=4):
    frames = np.asarray(frames,np.float32).reshape(-1,268)
    valid = np.asarray(valid,bool).reshape(-1) & np.isfinite(frames).all(axis=1)
    usable = bool(valid.sum() >= min_valid_frames)
    baseline = np.median(frames[valid],axis=0) if usable else np.zeros(268,np.float32)
    scale = max(float(np.quantile(np.abs(frames[valid]-baseline),.995)),1.) if usable else 1.
    return baseline.reshape(1,1,268).astype(np.float32), np.array([scale],np.float32), np.array([usable])


def o10_contact_fields(taxels, sample_valid, baseline, scale, baseline_valid,
                       *, calibrated=True, off_threshold=1., on_threshold=3.):
    """One native sensor: [1,T,1,268]. Labels are noise-relative weak labels."""
    if not 0 <= off_threshold < on_threshold:
        raise ValueError('Invalid contact thresholds')
    taxels = np.asarray(taxels,np.float32)
    if taxels.ndim != 4 or taxels.shape[0] != 1 or taxels.shape[2:] != (1,268):
        raise ValueError('O10 history must have shape [1,T,1,268]')
    if not calibrated:
        baseline, scale, baseline_valid = np.zeros_like(baseline), np.ones_like(scale), np.zeros_like(baseline_valid)
    valid = np.asarray(sample_valid,bool) & np.isfinite(taxels).all(axis=(-2,-1)) & baseline_valid[:,None] & calibrated
    taxels = np.where(valid[:,:,None,None],taxels,0).astype(np.float32)
    current_valid = valid[:,-1]
    # O10 contacts occupy only a few channels. A mean over all 268 would dilute
    # observed contacts below threshold; average the four largest deviations.
    deviation = np.abs((taxels[:,-1]-baseline)/scale[:,None,None]).reshape(1,268)
    score = np.partition(deviation,-4,axis=-1)[:,-4:].mean(axis=-1)
    positive = np.any(current_valid & (score >= on_threshold))
    negative = np.all(current_valid) and np.all(score <= off_threshold)
    return {'tactile':taxels, 'tactile_valid':current_valid, 'tactile_frame_valid':valid.any(axis=0),
            'tactile_sample_valid':valid, 'tactile_baseline':np.asarray(baseline,np.float32),
            'tactile_scale':np.asarray(scale,np.float32), 'tactile_contact':np.asarray(positive,np.float32),
            'tactile_contact_valid':np.asarray(positive or negative,bool)}

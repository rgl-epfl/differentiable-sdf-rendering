import fastsweep


def redistance(phi, method='fastsweep'):

    if method == 'fastsweep':
        return fastsweep.redistance(phi)
    elif method == 'fmm':
        import skfmm
        import numpy as np
        return skfmm.distance(phi, dx=1 / np.array(phi.shape))
    else:
        raise ValueError("Invalid re-distancing method")

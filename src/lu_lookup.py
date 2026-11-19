import numpy as np
Y0,X0=24*240,122*160
_F=None
def feats(lat,lon):
    global _F
    if _F is None: _F=dict(np.load('hn/lu_features.npz'))
    gy=(np.floor(np.asarray(lat)*240)-Y0).astype(int); gx=(np.floor(np.asarray(lon)*160)-X0).astype(int)
    return {k:v[gy,gx] for k,v in _F.items()}

"""
Check nans
"""
import numpy as np
from .common import subplotLabel, getSetup
from msresist.validations import preprocess_ebdt_mcf7
from ..clustering import MassSpecClustering

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    _, f = getSetup((15, 14), (4, 3), multz={3: 1, 10: 1})

    x = preprocess_ebdt_mcf7()
    d = x.select_dtypes(include=['float64']).T
    i = x.select_dtypes(include=['object'])

    pam_model = MassSpecClustering(i, ncl=20, SeqWeight=10, distance_method="PAM250").fit(d, "NA")
    assert np.all(np.isfinite(pam_model.transform()))

    return f
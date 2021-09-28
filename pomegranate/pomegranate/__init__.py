# __init__.py: pomegranate
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


"""
For detailed documentation and examples, see the README.
"""

import os

from .base import *

from .distributions import *
from .kmeans import Kmeans
from .gmm import GeneralMixtureModel

__version__ = '0.14.5'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def from_json(s):
    """A robust loading method.

    This method can load an appropriately formatted JSON object from any model
    in pomegranate and return the appropriate object. This relies mostly on the
    'class' attribute in the JSON.

    Parameters
    ----------
    s : str
            Either the filename of a JSON object or a string that is JSON formatted,
            as produced by any of the `to_json` methods in pomegranate.
    """

    try:
        d = json.loads(s)
    except BaseException:
        try:
            with open(s, 'r') as f:
                d = json.load(f)
        except BaseException:
            raise IOError("String must be properly formatted JSON or filename of properly formatted JSON.")

    if d['class'] == 'Distribution':
        return Distribution.from_json(s)
    elif d['class'] == 'GeneralMixtureModel':
        return GeneralMixtureModel.from_json(s)
    else:
        raise ValueError("Must pass in an JSON with a valid model name.")

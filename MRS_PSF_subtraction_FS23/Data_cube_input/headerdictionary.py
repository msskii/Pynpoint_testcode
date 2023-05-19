"""
Header saved as a list converted to a dictionary

"""

from typeguard import typechecked

from typing import List, Dict




@typechecked
def header_to_dict(header: List[str]) -> Dict:
    """
    Function which converts a list of header information with entries 'key = value'
    or ' key = [value1,…]' into a dictionary

    Parameters
    ----------
    header : List[str]
        Header information from the FITS file that is read.

    Returns
    -------
    Dict
        """

    header_dict = {}
    for keyval_pair in header:
        key, value = keyval_pair.split(" = ")
        # if isinstance(value,bytes) and value[0]
        header_dict[key] = value
    return header_dict


def edge_weight_to_string(weight: {int, float}) -> str:
    """
    returns a numeric edge weight as an appropriately formatted string

    :param      weight:  The edge weight to convert to string.
    :type       weight:  int or float

    :returns:   properly formatted weight string
    :rtype:     string
    """
    if isinstance(weight, int):
        wt_str = '{weight:d}'.format(weight=weight)
    elif isinstance(weight, float):
        wt_str = '{weight:.{digits}f}'.format(weight=weight,
                                              digits=2)

    return wt_str

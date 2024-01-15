import numpy as np
from numpy.linalg import norm
from math import floor, ceil

def integers_between(a, b):
    """
    Computes all integers between `a` and `b`.
    """
    
    if a < b:
        return np.arange(floor(a)+1, floor(b)+1)
    else:
        return np.arange(ceil(a)-1, ceil(b)-1, -1)


def get_box_crossings(dest, origin, boxsize): 
    """
    Given a ray traveling between two points, finds the intersection points of
    a 3D grid of integer multiples of boxsize.

    Returns
    -------
    all_edges: (N,3) array
        The list of all crossed bin edges from origin to destination, and the
        destination coordinates.
    all_edge_dists: (N,) array
        The distances from origin to each point in all_edges.
    all_box_gridcoords: (N,3) array
        The list of the grid coordinates of the bins that the ray travels through.
    """
    
    edges_list = [dest]
    edge_dists_list = [norm(np.atleast_2d(dest-origin), axis=1)]
    
    for i in range(3): #loop over dimensions, get all edge crossings in each dimension
        x_f = dest[i]
        x_0 = origin[i]
        
        x_crossings = boxsize*integers_between(x_0/boxsize, x_f/boxsize)
        if len(x_crossings)==0:
            continue
        m = (dest - origin)/(x_f - x_0)
        # y = y_0 + m(x-x_0)
        edges = origin + np.atleast_2d(m) * np.atleast_2d(x_crossings - x_0).T

        edges_list.append(edges)
        edge_dists_list.append(norm(edges - origin, axis=1))
    
    all_edge_dists, index_arr = np.unique(np.concatenate(edge_dists_list), return_index=True)
    all_edges = np.vstack(edges_list)[index_arr]
    
    all_midpoints = (np.vstack((np.atleast_2d(origin), all_edges[:-1])) + all_edges)/2
    all_box_gridcoords = (all_midpoints // boxsize).astype(int)

    return all_edges, all_edge_dists, all_box_gridcoords

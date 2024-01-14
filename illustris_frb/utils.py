import numpy as np
from numpy.linalg import norm

def signed_range(q):
    if q >= 0:
        return np.arange(q)
    else:
        return np.arange(0, q, -1)

def get_box_crossings(dest, origin, boxsize): 
    
    edges_list = [dest]
    edge_dists_list = [norm(np.atleast_2d(dest-origin), axis=1)]
    
    for i in range(3): #loop over dimensions, get all edge crossings in each dimension
        x_f = dest[i]
        x_0 = origin[i]
        if (x_f//boxsize) == (x_0//boxsize): #no edge crossings
            continue
            
        unit_vec_i = (dest - origin)/abs(x_f - x_0)
        edge_1 = origin + unit_vec_i * (boxsize - x_0%boxsize)
        
        q = abs(x_f - edge_1[i]) // boxsize + 1 # no. of total edge crossings in the x-direction
        
        edges = edge_1 + np.atleast_2d(unit_vec_i) * boxsize * np.atleast_2d(signed_range(q)).T
        edges_list.append(edges)
        edge_dists_list.append(norm(edges - origin, axis=1))
    
    all_edge_dists, index_arr = np.unique(np.concatenate(edge_dists_list), return_index=True)
    all_edges = np.vstack(edges_list)[index_arr]
    all_box_gridcoords = (np.vstack((np.atleast_2d(origin), all_edges[:-1])) / boxsize).astype(int)
    
    return all_edges, all_edge_dists, all_box_gridcoords
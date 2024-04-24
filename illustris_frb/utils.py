import numpy as np
from numpy.linalg import norm
from math import floor, ceil

def integers_between(a, b):
    """
    Computes all integers on [a, b).
    """
    
    if a < b:
        return np.arange(floor(a)+1, floor(b)+1)
    else:
        return np.arange(ceil(a)-1, ceil(b)-1, -1)

def index_to_coord(index, n_bins):
    
    res = np.zeros(3, dtype=int)
    for i in range(3):
        q, index = divmod(index, n_bins**(2-i))
        res[i] = q
    return res.astype(int)

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
    dest = np.asarray(dest)
    origin = np.asarray(origin)
    
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

def unit_sphere(N):
    pts = np.random.normal(size=(N, 3))
    norms = norm(pts, axis=1)
    mask = norms.nonzero()
    return pts[mask] / np.atleast_2d(norms[mask]).T

def is_within_cone(theta, phi, theta_0, phi_0, conesize):
    #from spherical law of cosines
    return np.arccos(np.cos(theta_0)*np.cos(theta) + 
                     np.sin(theta_0)*np.sin(theta)*np.cos(phi-phi_0)) < conesize

def rotate(theta1, phi1, dtheta, dphi):
    """
    Converts between two spherical coordinate systems, (theta0, phi0) and
    (theta1, phi1). The forward transformation (0 to 1) occurs by rotating
    azimuthally by -dphi, then transforming into a system with a pole at 
    (dtheta, np.pi/2).
    """

    theta0 = np.arccos( np.cos(theta1)*np.cos(dtheta) - np.sin(theta1)*np.sin(dtheta)*np.sin(phi1) )
    phi0 = np.arccos( np.sin(theta1)*np.cos(phi1) / np.sin(theta0) ) * \
           np.sign( ( np.cos(theta1)*np.sin(dtheta) + np.sin(theta1)*np.cos(dtheta)*np.sin(phi1) ) / np.cos(theta0) )

    return theta0, np.mod(phi0+dphi, 2*np.pi)

# def unpack_lims(lims, res):
#     x_min = lims[0][0]
#     x_max = lims[0][1]
#     y_min = lims[1][0]
#     y_max = lims[1][1]
    
#     n2 = int((x_max-x_min)/res) * int((y_max-y_min)/res)
    
#     return x_min, x_max, y_min, y_max, n2

# def coord2flatpix(x, y, lims, res, scale=1): 
#     #assumes positive integer res and bin edges --> use scale
#     #input x and y must be between these limits
    
#     lims_ = (np.asarray(lims)*scale).astype(int)
#     res_ = int(res*scale)
    
#     x_min, x_max, y_min, y_max, n2 = unpack_lims(lims_, res_)
    
#     n2 = (x_max-x_min)//res_ 
    
#     return ((x*scale - x_min)//(res_))**n2 + \
#            ((y*scale - y_min)//(res_))
    
# def fast_voxel_3d(u, v, t_f):
#     old_settings = np.seterr(divide='ignore')

#     X, Y, Z = u
#     stepX, stepY, stepZ = np.sign(v)
#     tDeltaX, tDeltaY, tDeltaZ = np.abs(norm(v)/v)
#     tMaxX, tMaxY, tMaxZ = np.abs(((np.floor(u*np.sign(v))+1)*np.sign(v) - u)*tDelta)

#     t = 0
#     res = []
#     while (t < t_f):
#         res.append((X,Y,Z))
#         if tMaxX < tMaxY:
#             if tMaxX < tMaxZ:
#                 X += stepX
#                 t = tMaxX
#                 tMaxX += tDeltaX
#             elif tMaxX > tMaxZ:
#                 Z += stepZ
#                 t = tMaxZ
#                 tMaxZ += tDeltaZ
#             else:
#                 X, Z = X+stepX, Z+stepZ
#                 t = tMaxX
#                 tMaxX += tDeltaX
#                 tMaxZ += tDeltaZ
#         elif tMaxX > tMaxY:
#             if tMaxY < tMaxZ:
#                 Y += stepY
#                 t = tMaxY
#                 tMaxY += tDeltaY
#             elif tMaxY > tMaxZ:
#                 Z += stepZ
#                 t = tMaxZ
#                 tMaxZ += tDeltaZ
#             else:
#                 Y, Z = Y+stepY, Z+stepZ
#                 t = tMaxY
#                 tMaxY += tDeltaY
#                 tMaxZ += tDeltaZ
#         else:
#             if tMaxX < tMaxZ:
#                 X, Y = X+stepX, Y+stepY
#                 t = tMaxX
#                 tMaxX += tDeltaX
#                 tMaxY += tDeltaY
#             elif tMaxX > tMaxZ:
#                 Z += stepZ
#                 t = tMaxZ
#                 tMaxZ += tDeltaZ
#             else:
#                 X, Y, Z = X+stepX, Y+stepY, Z+stepZ
#                 t = tMaxX
#                 tMaxX += tDeltaX
#                 tMaxY += tDeltaY
#                 tMaxZ += tDeltaZ

#     np.seterr(**old_settings)
#     return np.array(res)
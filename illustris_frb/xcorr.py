import numpy as np
from numpy.linalg import norm
from .utils import isSquare

"""
The 2D flat sky cross-power spectrum
--------------------------------------------------------------------------------

Let us call the input array $a(\vec{m})$. If it has dimension $(N_0, N_1)$, then
$m = (m_0, m_1)$ with $m_0 \in \{0, \ldots, N_0-1\}$ and
$m_1 \in \{0, \ldots, N_1-1\}$.

The 2D Fourier transform (as implemented in `numpy`) is defined as 
$$A(\vec{n}) = A_{n_0, n_1} = \sum_{m_0, m_1} a_{m_0, m_1} \exp\left(-2\pi i\left(
\frac{n_0}{N_0}m_0 + \frac{n_1}{N_1}m_1\right)\right)$$
with $n_0 \in \{0, \ldots, N_0-1\}$ and $n_1 \in \{0, \ldots, N_1-1\}$.

This is the output of `np.fft.fft2`. At this point it is helpful to shift the
$n\text{s}$ such that $n = 0, 1, \ldots, -n/2, \ldots, -1$, so that we can
account for the positive and negative frequencies.

After shifting, the corresponding wavenumber to some $n$ is $\frac{2\pi n}{N}$.
Because $\vec{\theta} = s\vec{m}$, we scale $n$ by the pixel size $s$ to get
$\ell$,
$$\vec{\ell} = \frac{2\pi}{s}\left(\frac{n_0}{N_0},\frac{n_1}{N_1}\right)  $$

Therefore, if we have $\vec{\theta} = s\vec{m}$ (up to a constant), then
$$A(\vec{\ell}) = A(\vec{n}(\vec{\ell})) = \sum_{\vec{\theta}} a(\vec{\theta})
\exp(-i\vec{\ell} \cdot \vec{\theta}) $$

To compute the cross correlation coefficients (see my
[notes](https://www.overleaf.com/project/65da4664effb352b8f6f9b13)):
$$C_\ell^{ab} = \frac{s^2}{N_0 N_1}\langle A(\vec{\ell})^* B(\vec{\ell}^\prime)
\rangle = \frac{s^2}{N_0 N_1}\frac{1}{\mathrm{\#\, of |\vec{\ell}|=\ell}}
\sum_{|\vec{\ell}|=\ell} A(\vec{\ell})^* B(\vec{\ell}) $$
"""

def cross_power_estimator_2d(arr_a, arr_b, s, delta_ell, ell_max=None):
    """
    Calculates the 2D cross-power spectrum for two 2D arrays.

    Parameters:
    ----------
    arr_a: ((M,N) arr)
        The first input array.
    arr_b: ((M,N) arr)
        The second input array.
    s: float
        The pixel size.
    delta_ell: float
        The bin size for the power spectrum.
    ell_max: float, optional
        The maximum value of the wavenumber. If not provided, it is set to the
        maximum wavenumber in the grid.
    """

    N_0, N_1 = arr_a.shape
    
    f_a = np.fft.fft2(arr_a)
    f_b = np.fft.fft2(arr_b) 
    f_ab = np.real(np.conjugate(f_a)*f_b)
    
    # scale
    vec_ells = np.indices((N_0, N_1), dtype=float)
    for i, N in zip((0,1), (N_0, N_1)): 
        vec_ells[i][ vec_ells[i] >= np.ceil(N/2) ] -= N #effectively, this is fftshift
        vec_ells[i] /= N
    vec_ells = 2*np.pi/s * vec_ells
    ells = norm(vec_ells, axis=0)
    
    if ell_max is None:
        ell_max = np.pi/s
    ell_bin_edges = np.arange(0, ell_max, delta_ell)
    counts, _ = np.histogram(ells.flatten(), bins=ell_bin_edges)
    tot, _ = np.histogram(ells.flatten(), bins=ell_bin_edges, weights=f_ab.flatten())

    ell_mids = ell_bin_edges[:-1]+delta_ell/2
    res = (s**2 / (N_0 * N_1)) * tot/counts
    mask = np.nonzero(counts)
    
    return ell_mids[mask], res[mask]

def get_ells(shape, s, nbins):
    """
    Bins the ells for a square array.
    """
    N = shape[0]
    vec_ells = np.indices(shape, dtype=float)
    vec_ells[ vec_ells >= np.ceil(N/2) ] -= N #effectively, this is fftshift
    vec_ells = 2*np.pi/(N*s) * vec_ells
    ells = norm(vec_ells, axis=0)
    
    ell_bin_edges = np.logspace(np.log10(2*np.pi/(N*s)), np.log10(2*np.pi/s), nbins+1)
    delta_ells = np.ediff1d(ell_bin_edges)
    ell_mids = ell_bin_edges[:-1]+delta_ells/2
    counts, _ = np.histogram(ells.flatten(), bins=ell_bin_edges)
    mask = counts > 0

    return ell_mids, delta_ells, ell_bin_edges, ells, counts, mask

def cross_power_estimator(arr_a, arr_b, s=0.0008, nbins=50, return_deltas=False):
    """
    Calculates the N-D cross-power spectrum for two N-D square arrays.
    """

    if arr_a.shape != arr_b.shape:
        raise Exception('Input arrays must have the same size')
    if not isSquare(arr_a.shape):
        raise Exception('Input arrays must be square')
    
    N = arr_a.shape[0]
    ell_mids, delta_ells, ell_bin_edges, ells, counts, mask = get_ells(
        arr_a.shape, s, nbins)
    
    f_a = np.fft.fftn(arr_a)
    f_b = np.fft.fftn(arr_b) 
    f_ab = np.real(np.conjugate(f_a)*f_b)
    
    tot, _ = np.histogram(ells.flatten(), bins=ell_bin_edges, weights=f_ab.flatten())
    tot, counts, ell_mids, delta_ells = (
        tot[mask], counts[mask], ell_mids[mask], delta_ells[mask])

    res = (s/N)**(arr_a.ndim) * tot/counts
    if return_deltas:
        return ell_mids, res, delta_ells
    return ell_mids, res

def get_Clerr_from_Cls(DMs, N_gs, ell_mids, delta_ells, Clgg, ClDD, res=0.0008):
    """
    Get the Gaussian error bar for a cross-power spectrum from the auto power spectrum.
    """
    Omega = (len(DMs)*res)**2
    NlDg2 = (Clgg + 1/(np.sum(N_gs)/Omega))*(ClDD + (np.var(DMs)*res**2))
    return 1 /  np.sqrt(Omega * ell_mids * delta_ells/(2*np.pi) / NlDg2)

def get_Clerr(DMs, N_gs, s=0.0008, nbins=100):
    """
    Compute the Gaussian error bar for the cross-correlation of two fields, the
    second of which is an overdensity field.
    """
    delta_gs = N_gs/np.mean(N_gs) - 1
    ell_mids, Clgg = cross_power_estimator(delta_gs, delta_gs, s, nbins)
    ell_mids, ClDD, delta_ells = cross_power_estimator(DMs, DMs, s, nbins, True)

    return get_Clerr_from_Cls(DMs, N_gs, ell_mids, delta_ells, Clgg, ClDD, s)

# cross power spectrum with incomplete data using the optimal quadratic estimator
# see https://www.overleaf.com/read/vdpbdpjvwrmp#2e27ab

def cross_oqe(DM, delta_g, frb_mult, s=0.0008, nbins=20, return_deltas=False):
    """
    Computes the DM-galaxy cross-correlation using the optimal quadratic
    estimator for an imcomplete DM field. Assumes uncorrelated variance. 

    Parameters
    ----------
    DM : (N,N) arr
        DM field 
    delta_g : (N,N) arr
        Galaxy overdensity field
    frb_mult : (N,N) arr
        Number of FRBs in each pixel
    s : float
        Pixel size in radians
    delta_ell : int
        Power spectrum binsize
    """

    if DM.shape != delta_g.shape:
        raise Exception('Input arrays must have the same size')
    if not isSquare(DM.shape):
        raise Exception('Input arrays must be square')

    N = len(DM)
    A = (N*s)**2
    ell_mids, delta_ells, ell_bin_edges, ells, counts, mask = get_ells(
        DM.shape, s, nbins)

    where_frb = frb_mult > 0
    n = np.sum(where_frb)
    varD, varG = np.var(DM[where_frb]), np.var(delta_g)
    invD, invG = where_frb.astype(int) / varD, np.ones_like(frb_mult) / varG

    ## compute numerator
    Dd = (DM - np.mean(DM)) * invD
    Gg = delta_g * invG
    num = np.conjugate(np.fft.fft2(Dd)) * np.fft.fft2(Gg) / A
    num_l, _ = np.histogram(ells.flatten(), bins=ell_bin_edges, weights=num.flatten()) 
    num_l, counts, ell_mids, delta_ells = (
        num_l[mask], counts[mask], ell_mids[mask], delta_ells[mask])

    ## compute Fisher matrix
    Finv = (2*N**2*s**4*varD*varG) / (counts*n) # diagonal matrix
    ClDg = Finv * np.real(num_l) / 2

    if return_deltas:
        return ell_mids, ClDg, delta_ells
    return ell_mids, ClDg
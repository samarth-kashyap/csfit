# {{{ Library imports
import time
import numpy as np
from math import pi
from math import sqrt
from astropy.io import fits
import matplotlib.pyplot as plt
from heliosPy import datafuncs as cdata
from numpy.polynomial.legendre import legval
# }}} Library imports


# {{{ Global variables
mode_data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')
# }}} global vars


# {{{ class coupMat
class coupMat:
    """ Class for handling objects related to C_{ij} (parameter matrix)

    Attributes:
    -----------
    lmin - int
        minimum value of ell for which C_{ij} if fit
    lmax - int
        maximum value of ell for which C_{ij} is fit
    dl_coup - int
        maximum value of abs(i - j); i^th mode is coupled to j^th mode
        if |i - j| <= dl_coup
    msz_coup - int
        size of coupling matrix C_{ij}; matrix is square
    coup_mat - np.ndarray(ndim=2, dtype=float)
        the coupling matrix C_{ij}
    la_mat - np.ndarray(ndim=2, dtype=int)
        (a : above) matrix containing ell values of upper index i of C^i_j
    lb_mat - np.ndarray(ndim=2, dtype=int)
        (b : below) matrix containing ell values of lower index j of C^i_j
    ind_mat - np.ndarray(ndim=2, dtype=int)
        matrix containing a combined index of (i, j)
    mask - np.ndarray(ndim=2, dtype=bool)
        mask to filter out values beyond dl_coup

    Methods:
    --------
    generate_synth()
        generates synthetic C^i_j for testing

    """

    def __init__(self, lmin, lmax, dl_coup):
        self.lmin = lmin
        self.lmax = lmax
        self.dl_coup = dl_coup
        self.msz_coup = lmax - lmin
        self.coup_mat = np.identity(self.msz_coup, dtype=float)

        bool_mat = np.zeros((self.msz_coup, self.msz_coup), dtype=np.bool)
        ell1_mat = np.zeros((self.msz_coup, self.msz_coup), dtype=np.int)
        ell2_mat = np.zeros((self.msz_coup, self.msz_coup), dtype=np.int)
        for i in range(self.msz_coup):
            for j in range(self.msz_coup):
                if abs(i-j) <= self.dl_coup:
                    bool_mat[i, j] = True
                    ell1_mat[i, j] = i + lmin
                    ell2_mat[i, j] = j + lmin
        ind_mat = np.zeros_like(bool_mat, dtype=np.int)
        ind_mat[bool_mat] = np.arange(bool_mat.sum())

        self.la_mat = ell1_mat
        self.lb_mat = ell2_mat
        self.ind_mat = ind_mat
        self.mask = bool_mat
        return None

    def generate_synth(self):
        """Generates synthetic C^i_j

        Block diagonal C^i_j which is diagonally dominant assuming
        that the self coupling is the strongest.

        Inputs:
        -------
        None

        Outputs:
        --------
        None

        *Updates the coup_mat attribute of the coupMat object.*

        """

        self.coup_mat[self.mask] = 0.3*np.random.rand(self.mask.sum())
        for i in range(self.msz_coup):
            self.coup_mat[i, i] = 1 - 0.3*np.random.rand()
            _norm = sqrt((abs(self.coup_mat[i, :])**2).sum())
            self.coup_mat[i, :] /= _norm
        return None

    def cij_val(self, la, lb):
        """Returns the value of coupling matrix for given la and lb

        Inputs:
        -------
        la - int
            spherical harmonic degree (upper)
        lb - int
            spherical harmonic degree (lower)

        Returns:
        --------
        cij(la, lb) - complex
            the coupling matrix element
        """
        mask_la = self.la_mat == la
        mask_lb = self.lb_mat == lb
        mask_ij = mask_la * mask_lb
        cij = self.coup_mat[mask_ij]
        try:
            return cij[0]
        except IndexError:
            return 0.0
# }}} coupMat


# {{{ def resid2(x, f1N, p1, noise_level)
def resid2(x, f1N, p1, noise_level):
    f1N_fit, _temp = trans2(x, p1, noise_level)
    diff = f1N_fit - f1N
    resid = (abs(diff)**2).sum()
    return f1N_fit, diff, resid
# }}} resid2(x, f1N, p1, noise_level)


# {{{ def gauss_newton(J, res, p1, damping)
def gauss_newton(J, res, p1, damping):
    jtj = J.transpose().dot(J)
    jtji = np.linalg.pinv(jtj, rcond=1e-15)
    p_add = jtji.dot(J.transpose().dot(res))
    return p1 + damping*p_add
# }}} gauss_newton(J, res, p1, damping)


# {{{ iter_gn2(p1, x, f1N, Niter, damping, noise_level)
def iter_gn2(p1, x, f1N, Niter, damping, noise_level):
    for i in range(Niter):
        J = jacob2(len(p1), len(x), x, p1)
        f1N_fit, res_vec, S = resid2(x, f1N, p1, noise_level)
        if i > 0 and S > sval[-1]:
            dp = -damping
        else:
            dp = damping
        p1 = gauss_newton(J, res_vec, p1, dp)

        sval = S if i == 0 else np.append(sval, S)
    return pval0, pval1, svaldef 
# }}} iter_gn2(p1, x, f1N, Niter, damping, noise_level)


# {{{ def finda1(data, l, n, m)
def finda1(data, l, n, m):
    """
    Find a coefficients for given l, n, m
    """
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) * (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l}, n = {n}")
        return None, None, None
    splits = np.append([0.0], data[modeindex, 12:48])
    totsplit = legval(1.0*m/L, splits)*L
    return totsplit
# }}} finda1(data, l, n, m)


# {{{ def create_synth(l1, n1, l2, n2, coupmat)
def create_synth(l1, n1, l2, n2, coupmat):
    """Create synthetic dataset for testing out fitting algorithm
    Inputs:
    -------

    Outputs:
    --------

    """
    # calculation parameters
    dl = 6              # delta_ell for leakage matrix
    dm = 15             # delta_m for leakage matrix
    dl_mat = 6      # max delta_ell for leakage matrix
    dm_mat = 15     # max delta_m for leakage matrix

    rsun = 6.9598e10
    twopiemin6 = 2*pi*1e-6

    daynum = 1          # length of time series
    tsLen = 138240  # array length of the time series

    # time domain and frequency domain
    t = np.linspace(0, 72*24*3600*daynum, tsLen*daynum)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(t.shape[0], dt)*1e6
#    df = freq[1] - freq[0]  # frequencies in microHz

    rleaks = fits.open('/home/g.samarth/leakage/rleaks1.fits')[0].data
    horleaks = fits.open('/home/g.samarth/leakage/horleaks1.fits')[0].data
    rleaks1 = rleaks[:, :, l1, :].copy()
    rleaks2 = rleaks[:, :, l2, :].copy()
    horleaks1 = horleaks[:, :, l1, :].copy()
    horleaks2 = horleaks[:, :, l2, :].copy()
    del rleaks, horleaks

    # considering only frequencies in a window
    # where cross spectrum is significant
    cenfreq, cenfwhm, cenamp = cdata.findfreq(mode_data, l1, n1, 0)
    indm = cdata.locatefreq(freq, cenfreq - l1*0.6)
    indp = cdata.locatefreq(freq, cenfreq + l1*0.6)
    freq = freq[indm:indp].copy()

    cs = np.zeros((l1+1, freq.shape[0]), dtype=complex)
    phi1sum = np.zeros(freq.shape[0], dtype=complex)

    # new norms
    cnl1 = np.loadtxt(writedir + f"norm_{l1:03d}_{n1:02d}")
    cnl2 = np.loadtxt(writedir + f"norm_{l2:03d}_{n2:02d}")
    omeganl1, fwhmnl1, amp1 = cdata.findfreq(mode_data, l1, n1, 0)
    omeganl2, fwhmnl2, amp2 = cdata.findfreq(mode_data, l2, n2, 0)

    norm1 = amp1**2 * omeganl1 * omeganl1 * cnl1 * fwhmnl1 * twopiemin6**3
    norm2 = amp2**2 * omeganl2 * omeganl2 * cnl2 * fwhmnl2 * twopiemin6**3
    norm = sqrt(norm1*norm2)

    t1 = time.time()
    for m1 in range(l1+1):
        for dell1 in range(-dl, dl+1):
            dell2 = dell1 - l2 + l1
            l1p = l1 + dell1
            l2p = l1p
#            cij = 1.0  # 0.8 + 0.2*np.random.rand()
            cij1 = coupmat.cij_val(l1, l1p)
            cij2 = coupmat.cij_val(l2, l2p)
            for dem in range(-dm, dm+1):
                m1p = m1 + dem
                if (abs(m1p) <= l1p):
                    omeganl1p, fwhmnl1p, amp1p = cdata.findfreq(mode_data,
                                                                l1p, n1, m1p)
                    mix = (274.8*1e2*(l1p+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1p**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m1+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m1+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m1+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m1+249]
                    phi1p = cdata.lorentzian(omeganl1p, fwhmnl1p, freq)
                    phi1sum += (leak1*leak2 * phi1p*phi1p.conjugate() *
                                cij1*cij2)
        cs[m1, :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print(f"Time taken for computation (positive m) = " +
          f"{(t2 - t1):6.2f} seconds")
    return cs
    # }}} create_synth(l1, n1, l2, n2, coupmat)


if __name__ == "__main__":
    print(f"program start")

    # directories
    writedir = '/scratch/g.samarth/csfit/'

    coupmat = coupMat(194, 206, 6)
    coupmat.generate_synth()
    l1, n1, l2, n2 = 200, 0, 202, 0
    cs = create_synth(l1, n1, l2, n2, coupmat)

    plt.figure()
    im = plt.imshow(abs(cs), aspect=cs.shape[1]/cs.shape[0],
                    vmax=abs(cs).max()/5, cmap="Greys")
    plt.colorbar(im)
    plt.show()

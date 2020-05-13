import numpy as np
from math import sqrt
from numpy.polynomial.legendre import legval
from astropy.io import fits
from math import pi
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("/home/g.samarth/")
from heliosPy import datafuncs as cdata


class coupmat:
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

    def generate_synth(self):
        self.coup_mat[self.mask] = 0.3*np.random.rand(self.mask.sum())
        for i in range(self.msz_coup):
            self.coup_mat[i, i] = 1 - 0.3*np.random.rand()
            _norm = sqrt((abs(self.coup_mat[i, :])**2).sum())
            self.coup_mat[i, :] /= _norm
        return None


def resid2(x, f1N, p1, noise_level):
    f1N_fit, _temp = trans2(x, p1, noise_level)
    diff = f1N_fit - f1N
    resid = (abs(diff)**2).sum()
    return f1N_fit, diff, resid


def gauss_newton(J, res, p1, damping):
    jtj = J.transpose().dot(J)
    jtji = np.linalg.pinv(jtj, rcond=1e-15)
    p_add = jtji.dot(J.transpose().dot(res))
    return p1 + damping*p_add


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
    return pval0, pval1, sval


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

    # new norms
    cnl1 = np.loadtxt(writedir + f"norm_{l1:03d}_{n1:02d}")
    cnl2 = np.loadtxt(writedir + f"norm_{l2:03d}_{n2:02d}")
    mode_data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')

    omeganl1, fwhmnl1, amp1 = cdata.findfreq(mode_data, l1, n1, 0)
    omeganl2, fwhmnl2, amp2 = cdata.findfreq(mode_data, l2, n2, 0)

    norm1 = amp1*amp1 * omeganl1*omeganl1 * cnl1 * fwhmnl1 * twopiemin6**3
    norm2 = amp2*amp2 * omeganl2*omeganl2 * cnl2 * fwhmnl2 * twopiemin6**3
    norm = sqrt(norm1*norm2)

    t1 = time.time()
    for m in range(l1+1):
        for dell1 in range(-dl, dl+1):
            dell2 = dell1 - l2 + l1
            l21 = l1 + dell1
            tempnorm = 1.0  # 0.8 + 0.2*np.random.rand()
            for dem in range(-dm, dm+1):
                m2 = m+dem
                if (abs(m2) <= l21):
                    omeganl1, fwhmnl1, amp1 = cdata.findfreq(mode_data,
                                                             l21, n2, m2)
                    mix = (274.8*1e2*(l21+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m+249]
                    phi1 = cdata.lorentzian(omeganl1 - finda1(mode_data,
                                                              l1, n1, m),
                                            fwhmnl1, freq)
                    phi1sum = phi1sum +\
                        leak1*leak2 * phi1*phi1.conjugate() * tempnorm
        cs[m, :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print(f" Total time taken = {(t2 - t1):10.5f} seconds")


if __name__=="__main__":
    print(f"program start")

    # directories
    writedir = '/scratch/g.samarth/csfit/'

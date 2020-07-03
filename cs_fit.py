# {{{ Library imports
import time
import numpy as np
from math import pi
from math import sqrt
from astropy.io import fits
import scipy.ndimage as sciim
import matplotlib.pyplot as plt
from globalvars import globalvars
from heliosPy import datafuncs as cdata
from numpy.polynomial.legendre import legval
# }}} Library imports


# {{{ Global variables
mode_data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')
gvar = globalvars()
twopiemin6 = gvar.twopiemin6
dm = gvar.dm
dl = gvar.dl
dm_mat = gvar.dm_mat
dl_mat = gvar.dl_mat
# }}} global vars


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

    # {{{ def __init__(self, lmin, lmax, dl_coup):
    def __init__(self, lmin, lmax, dl_coup):
        self.lmin = lmin
        self.lmax = lmax
        self.dl_coup = dl_coup
        self.msz_coup = lmax - lmin
        self.coup_mat = np.identity(self.msz_coup, dtype=complex)

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
        self.maxindex = self.mask.sum()
        return None
    # }}} __init__(self, lmin, lmax, dl_coup)

    # {{{ def generate_synth(self):
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
    # }}} generate_synth(self)

    # {{{ def cij_val(self, la, lb):
    def cij_val(self, la, lb):
        """Returns the value of coupling matrix for given la and lb

        Inputs:
        -------
        la - int
            (ell above) ell which is looped over
            spherical harmonic degree (upper)
        lb - int
            (ell below) target ell
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
    # }}} cij_val(self, la, lb)

    # {{{ def caddself, corr):
    def cadd(self, corr):
        assert len(corr) == self.maxindex
        cij = self.coup_mat.copy()
        print(f" Max correction to cij = {abs(corr).max()} ")
        cij[self.mask] += corr
        self.coup_mat = cij.copy()
        return None
    # }}} cadd(self, corr)


class spectra():
    """Class to handle synthetic spectral fitting

    """
    # {{{ def derotate(cs, l, n, freq, pm):
    def derotate(self, cs, l, n, freq, pm):
        """Derotate the given cross-spectra

        Inputs:
        -------
        cs - np.ndarray(ndim=2, dtype=complex)
            cross-spectra that needs to be derotated
        l - int
            spherical harmonic degree
        n - int
            radial order
        freq - np.ndarray(ndim=1, dtype=float)
            array containing the frequencies
        pm - int
            pm = 1  implies m >= 0
            pm = -1 implies m <= 0

        Outputs:
        --------
        cs_derot - np.ndarray(ndim=2, dtype=complex)
            derotated version of cs

        """
        L = sqrt(l*(l+1))
        data = np.loadtxt(gvar.leak_dir + 'hmi.6328.36')
        modeindex = np.where((data[:, 0] == l) *
                             (data[:, 1] == n))[0][0]
        splits = np.append([0.0], data[modeindex, 12:48])
        cs_derot = np.zeros(cs.shape, dtype=complex)
        df = (freq[1] - freq[0])*1e-6  # in Hz

        for m in range(l+1):
            delta_nu = legval(1.0*m/L, splits)*L  # in nHz
            delta_nu *= 1e-9
            const = -1 * pm * delta_nu / df
            _real = sciim.shift(cs[m, :].real, const, mode='wrap', order=1)
            _imag = sciim.shift(cs[m, :].imag, const, mode='wrap', order=1)
            cs_derot[m, :] = _real + 1j*_imag
        return cs_derot
    # }}} derotate(phi, l, n, freq, pm)

    # {{{ def resid_spectra(self):
    def resid_spectra(self):
        return self.cs_data - self.cs_synth
    # }}} resid_spectra(self)

    # {{{ def obj_spectra(self):
    def obj_spectra(self):
        diff = abs(self.resid_spectra())**2
        return diff.sum()
    # }}} obj_spectra(self)

    # {{{ def d_resid_spectra(self):
    def d_resid_spectra(self, index):
        bool_cij = self.cij.mask
        l1p = self.cij.la_mat[bool_cij][index]
        t = np.linspace(0, 72*24*3600, gvar.tsLen)
        freq = np.fft.fftfreq(t.shape[0], t[1] - t[0])*1e6
        indm = cdata.locatefreq(freq, self.cenfreq - self.l1*0.6)
        indp = cdata.locatefreq(freq, self.cenfreq + self.l1*0.6)
        freq = freq[indm:indp]

        d_cs = np.zeros((self.l1+1, freq.shape[0]), dtype=complex)
        csm = np.zeros((self.l1+1, freq.shape[0]), dtype=complex)

        # new norms
        cnl1 = np.loadtxt(writedir + f"norm_{self.l1:03d}_{self.n1:02d}")
        cnl2 = np.loadtxt(writedir + f"norm_{self.l2:03d}_{self.n2:02d}")
        omeganl1, fwhmnl1, amp1 = cdata.findfreq(mode_data, self.l1, self.n1, 0)
        omeganl2, fwhmnl2, amp2 = cdata.findfreq(mode_data, self.l2, self.n2, 0)

        norm1 = amp1**2 * omeganl1 * omeganl1 * cnl1 * fwhmnl1 * twopiemin6**3
        norm2 = amp2**2 * omeganl2 * omeganl2 * cnl2 * fwhmnl2 * twopiemin6**3
        norm = sqrt(norm1*norm2)

        t1 = time.time()
        dell1 = self.l1 - l1p
        dell2 = self.l2 - l1p
        for m1 in range(self.l1+1):
            m1p = np.arange(m1+gvar.dm, m1-gvar.dm-1, -1)
            cij1 = self.cij.cij_val(l1p, l1p)
            cij2 = self.cij.cij_val(l1p, l1p).conjugate()
            omeganl1p, fwhmnl1p, amp1p = cdata.findfreq_vecm(mode_data,
                                                            l1p, self.n1,
                                                            m1p)
            mix = (274.8*1e2*(l1p+0.5) /
                ((gvar.twopiemin6)**2*gvar.rsun))/omeganl1p**2
            leak1 = self.rleaks1[:, dell1+gvar.dl_mat, m1+249] +\
                mix*self.horleaks1[:, dell1+gvar.dl_mat, m1+249]
            leak2 = self.rleaks2[:, dell2+gvar.dl_mat, m1+249] +\
                mix*self.horleaks2[:, dell2+gvar.dl_mat, m1+249]
            phi1p = cdata.lorentzian_vec(omeganl1p, fwhmnl1p, freq)
            amp1p = amp1p[:, np.newaxis]
            d_cs[m1, :] = (leak1*leak2 @ (amp1p*amp1p*phi1p*phi1p.conjugate())*(cij1+cij2))
        t2 = time.time()

        d_cs *= norm

        # derotation of the synthetics
        d_cs = self.derotate(d_cs, self.l1, self.n1, freq, 1)

        # windowing (+/- 25 \micro Hz of cenfreq)
        pmfreq = 25
        indm = cdata.locatefreq(freq, self.cenfreq - pmfreq)
        indp = cdata.locatefreq(freq, self.cenfreq + pmfreq)
        freq = freq[indm:indp].copy()
        d_cs = d_cs[:, indm:indp].copy()
#        print(f"Time taken for computation (positive m) = " +
#              f"{(t2 - t1):6.2f} seconds")
        return d_cs.sum(axis=0)
    # }}} d_resid_spectra(self, l1p)

    # {{{ def grad_S(self):
    def grad_S(self):
        J = np.zeros((self.cij.maxindex, len(self.cs_synth)), dtype=np.float)
        for i in range(self.cij.maxindex):
            dr_dc = self.d_resid_spectra(i)
            J[i, :] = 2 * dr_dc
        self.J = J
        return J
    # }}} grad_S(self)

    # {{{ def hess_spectra(self):
    def hess_spectra(self):
        self.H = self.J @ self.J.transpose()
        return self.H
    # }}} hess_spectra(self)

    # {{{ def gauss_newton(self, dpg)
    def gauss_newton(self, dpg):
        self.res = self.resid_spectra()
        g = self.J @ self.res
        hi = np.linalg.pinv(self.H, rcond=1e-15)
        cij_corr = dpg * (hi @ g.transpose())
        self.cij.cadd(cij_corr)
        return self.cij
    # }}} gauss_newton(self, dpg)

    # {{{ def iter_GN(self, damping, Niter):
    def iter_GN(self, damping, Niter, reslimit=1e-10):
        self.count = 1
        res_list = np.zeros(Niter, dtype=np.float)
        for i in range(Niter):
            print(f" Iteration number = {i}/{Niter-1} ")
            self.cij = self.gauss_newton(damping)
            self.cs_synth = self.func_spectra(self.cij)
            self.count += 1
            S = self.obj_spectra()
            res_list[i] = S
        self.S_list = res_list
        return None
    # }}} iter_GN(self, damping, Niter)

    # {{{ def func_spectra(self, coupmat):
    def func_spectra(self, coupmat):
        t = np.linspace(0, 72*24*3600, gvar.tsLen)
        freq = np.fft.fftfreq(t.shape[0], t[1] - t[0])*1e6
        indm = cdata.locatefreq(freq, self.cenfreq - self.l1*0.6)
        indp = cdata.locatefreq(freq, self.cenfreq + self.l1*0.6)
        freq = freq[indm:indp]

        cs = np.zeros((self.l1+1, freq.shape[0]), dtype=complex)
        csm = np.zeros((self.l1+1, freq.shape[0]), dtype=complex)
        phi1sum = np.zeros(freq.shape[0], dtype=complex)

        # new norms
        cnl1 = np.loadtxt(writedir + f"norm_{self.l1:03d}_{self.n1:02d}")
        cnl2 = np.loadtxt(writedir + f"norm_{self.l2:03d}_{self.n2:02d}")
        omeganl1, fwhmnl1, amp1 = cdata.findfreq(mode_data, self.l1, self.n1, 0)
        omeganl2, fwhmnl2, amp2 = cdata.findfreq(mode_data, self.l2, self.n2, 0)

        norm1 = amp1**2 * omeganl1 * omeganl1 * cnl1 * fwhmnl1 * twopiemin6**3
        norm2 = amp2**2 * omeganl2 * omeganl2 * cnl2 * fwhmnl2 * twopiemin6**3
        norm = sqrt(norm1*norm2)

        t1 = time.time()
        for m1 in range(self.l1+1):
            m1p = np.arange(m1+gvar.dm, m1-gvar.dm-1, -1)
            for dell1 in range(-gvar.dl, gvar.dl+1):
                dell2 = dell1 - self.l2 + self.l1
                l1p = self.l1 - dell1
                cij1 = coupmat.cij_val(l1p, l1p)
                cij2 = coupmat.cij_val(l1p, l1p).conjugate()
                omeganl1p, fwhmnl1p, amp1p = cdata.findfreq_vecm(mode_data,
                                                                 l1p, self.n1,
                                                                 m1p)
                mix = (274.8*1e2*(l1p+0.5) /
                    ((gvar.twopiemin6)**2*gvar.rsun))/omeganl1p**2
                leak1 = self.rleaks1[:, dell1+gvar.dl_mat, m1+249] +\
                    mix*self.horleaks1[:, dell1+gvar.dl_mat, m1+249]
                leak2 = self.rleaks2[:, dell2+gvar.dl_mat, m1+249] +\
                    mix*self.horleaks2[:, dell2+gvar.dl_mat, m1+249]
                phi1p = cdata.lorentzian_vec(omeganl1p, fwhmnl1p, freq)
                amp1p = amp1p[:, np.newaxis]
                phi1sum += (leak1*leak2 @ (amp1p*amp1p*phi1p*phi1p.conjugate()) *
                            cij1*cij2)
            cs[m1, :] = phi1sum*norm
            phi1sum = 0.0*phi1sum
        t2 = time.time()
        cs = self.derotate(cs, self.l1, self.n1, freq, 1)
        print(f"Time taken for computation (positive m) = " +
              f"{(t2 - t1):6.2f} seconds")
#
        """
        for m1 in range(-self.l1, 0):
            m1p = np.arange(m1+gvar.dm, m1-gvar.dm-1, -1)
            for dell1 in range(-gvar.dl, gvar.dl+1):
                dell2 = dell1 - self.l2 + self.l1
                l1p = self.l1 - dell1
                cij1 = coupmat.cij_val(l1p, l1p)
                cij2 = coupmat.cij_val(l1p, l1p).conjugate()
                omeganl1p, fwhmnl1p, amp1p = cdata.findfreq_vecm(mode_data,
                                                                 l1p, self.n1,
                                                                 m1p)
                mix = (274.8*1e2*(l1p+0.5) /
                    ((gvar.twopiemin6)**2*gvar.rsun))/omeganl1p**2
                leak1 = self.rleaks1[:, dell1+gvar.dl_mat, m1+249] +\
                    mix*self.horleaks1[:, dell1+gvar.dl_mat, m1+249]
                leak2 = self.rleaks2[:, dell2+gvar.dl_mat, m1+249] +\
                    mix*self.horleaks2[:, dell2+gvar.dl_mat, m1+249]
                phi1p = cdata.lorentzian_vec(omeganl1p, fwhmnl1p, freq)
                amp1p = amp1p[:, np.newaxis]
                phi1sum += (leak1*leak2 @ (amp1p*amp1p*phi1p*phi1p.conjugate()) *
                            cij1*cij2)
            csm[abs(m1), :] = phi1sum*norm
            phi1sum = 0.0*phi1sum
        t2 = time.time()
        csm = self.derotate(csm, self.l1, self.n1, freq, -1)
        print(f"Time taken for computation (negative m) = " +
              f"{(t2 - t1):6.2f} seconds")
        """
#
        # restricting frequencies to within 25 \micro Hz of cenfreq
        pmfreq = 25
        indm = cdata.locatefreq(freq, self.cenfreq - pmfreq)
        indp = cdata.locatefreq(freq, self.cenfreq + pmfreq)
        freq = freq[indm:indp].copy()
        self.cs_synth = cs[:, indm:indp].sum(axis=0)
#        self.csm_synth = csm[:, indm:indp].sum(axis=0)
        return self.cs_synth  #, self.csm_synth
    # }}} func_spectra(self, coupmat)

    # {{{ def __init__(self, l1, n1, l2, n2, coupmat):
    def __init__(self, ln_data, coupmat):
        self.cij = coupmat
        self.l1, self.n1, self.l2, self.n2 = ln_data
        self.cenfreq, self.cenfwhm, cenamp = cdata.findfreq(mode_data,
                                                            self.l1, self.n1, 0)

        csd = np.load(f"{gvar.write_dir}csp_data_{self.n1:02d}"+
                      f"_{self.l1:03d}_{self.l2:03d}.npy")
        csm = np.load(f"{gvar.write_dir}csm_data_{self.n1:02d}"+
                      f"_{self.l1:03d}_{self.l2:03d}.npy")
        self.cs_data, self.csm_data = csd.sum(axis=0).copy(), csm.sum(axis=0)
        bg, bgm = self.cs_data[:50].mean(), self.csm_data[:50].mean()
        self.cs_data -= bg
        self.csm_data -= bgm
        del csd, csm, bg, bgm

        rleaks = fits.open(f"{gvar.leak_dir}rleaks1.fits")[0].data
        horleaks = fits.open(f"{gvar.leak_dir}horleaks1.fits")[0].data
        self.rleaks1 = rleaks[:, :, self.l1, :].copy()
        self.rleaks2 = rleaks[:, :, self.l2, :].copy()
        self.horleaks1 = horleaks[:, :, self.l1, :].copy()
        self.horleaks2 = horleaks[:, :, self.l2, :].copy()
        del rleaks, horleaks
        print(f"-----------------------------------------------\n" +
              f"       Initializing synthetic spectra          \n" +
              f"-----------------------------------------------\n")
        self.cs_synth = self.func_spectra(coupmat)
        self.res = self.resid_spectra()
        self.grad_S()
        self.hess_spectra()
        return None
    # }}} __init__(self, l1, n1, l2, n2, coupmat)


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
    # directories
    writedir = '/scratch/g.samarth/csfit/'

    coupmat = coupMat(194, 206, 0)
#    coupmat.generate_synth()
    ln_data = 200, 0, 200, 0
    CS = spectra(ln_data, coupmat)
    css_old = CS.cs_synth
    t1 = time.time()
    #CS.iter_GN(8e14, 3500)
    CS.iter_GN(0.1, 20)
    t2 = time.time()
    print(f"=================================================" +
          f"Total time taken = {(t2-t1)/60:5.2f} min")

    css = CS.cs_synth
    csd = CS.cs_data

    maxind = np.where(csd == np.amax(csd))[0][0]
    norm_old = csd.real[maxind]/css_old.real[maxind]
    norm = csd.real[maxind]/css.real[maxind]

    plt.figure(figsize=(15, 15))
    plt.plot(css.real * norm, 'r', label='synthetic - after fit')
    plt.plot(css_old.real * norm_old, '--', color='black', label='synthetic - before fit')
    plt.plot(csd.real, 'b', label='data')
    plt.legend()
    plt.savefig("/home/g.samarth/ps200fit.png")
    plt.show()

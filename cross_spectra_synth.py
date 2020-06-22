# {{{ Library imports
from numpy.polynomial.legendre import legval
#  Note that directory containing heliosPy needs to be added to PYTHONPATH
from heliosPy import datafuncs as cdata
import matplotlib.pyplot as plt
import scipy.ndimage as sciim
from astropy.io import fits
from math import sqrt, pi
import numpy as np
import argparse
import time
import sys
# }}} imports


# {{{ Global variables
# directories
writedir = '/scratch/g.samarth/csfit/'

# calculation parameters
dl = 6              # delta_ell for leakage matrix
dm = 15             # delta_m for leakage matrix
dl_mat = 6      # max delta_ell for leakage matrix
dm_mat = 15     # max delta_m for leakage matrix

rsun = 6.9598e10
twopiemin6 = 2*pi*1e-6

daynum = 1          # length of time series
tsLen = 138240  # array length of the time series

# reading mode parameters data
data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')
# }}} global vars


# {{{ Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--norms",
                    help="computes norms",
                    action="store_true")
parser.add_argument("--l",
                    help="spherical harmonic degree 1",
                    type=int)
parser.add_argument("--lp",
                    help="spherical harmonic degree 2",
                    type=int)
parser.add_argument("--n",
                    help="radial order 1", type=int)
parser.add_argument("--np",
                    help="radial order 2", type=int)
parser.add_argument("--plot",
                    help="plot result after computation",
                    action="store_true")
args = parser.parse_args()
# }}} args


# {{{ class multiline_plot():
class multiline_plot():
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
        self.lkterm = np.array([])

    def add_plot(self, plot_data, inc):
        _max = abs(plot_data).max()
        if _max > 0:
            plot_data /= _max
            plot_data += inc
        if abs(plot_data).sum() < 0.1:
            plot_data += inc
            self.ax.plot(plot_data, color='red')
        else:
            self.ax.plot(plot_data, color='black')

    def leak_terms(self, l1p, m1p, leak1, leak2, inc):
        if inc == 0:
            self.lkterm = np.append(self.lkterm, (l1p, m1p,
                                                  leak1, leak2))
        else:
            self.lkterm = np.vstack([self.lkterm, (l1p, m1p,
                                                   leak1, leak2)])
# }}} multiline_plot()


# {{{ def find_split(data, l, n, m):
def find_split(data, l, n, m):
    """ Find total frequency splitting (in nHz) using asymptotic polynomials

    Inputs:
    -------
    data - np.ndarray(ndim=2, dtype=float)
        contains the data from mode parameter file (e.g. hmi.6328.36)
    l - int
        spherical harmonic degree
    n - int
        radial order
    m - int
        azimuthal order

    Returns:
    --------
    totsplit - float
        frequency splitting in nHz

    Notes:
    ------
    Legendre polynomials are used to compute the splitting frequency.
    This is an asymptotic approximation to the general orthogonal
    polynomials as described in Schou?

    """
    L = sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0] == l) *
                             (data[:, 1] == n))[0][0]
    except IndexError:
        print(f"MODE NOT FOUND : l = {l}, n = {n}")
        return None, None, None
    splits = np.append([0.0], data[modeindex, 12:48])
    totsplit = legval(1.0*m/L, splits)*L
    return totsplit
# }}} find_split(data, l, n, m)


# {{{ def derotate(cs, l, n, freq, pm):
def derotate(cs, l, n, freq, pm):
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
    data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')
    cs_derot = np.zeros(cs.shape, dtype=complex)
    df = (freq[1] - freq[0])*1e-6  # in Hz

    for m in range(l+1):
        delta_nu = find_split(data, l, n, m)  # in nHz
        delta_nu *= 1e-9
#        const = -1*pm*delta_nu*72*24*3600
        const = -1 * pm * delta_nu / df
        _real = sciim.shift(cs[m, :].real, const, mode='wrap', order=1)
        _imag = sciim.shift(cs[m, :].imag, const, mode='wrap', order=1)
        cs_derot[m, :] = _real + 1j*_imag
    return cs_derot
# }}} derotate(phi, l, n, freq, pm)


# {{{ def plot_cs(cs, csm, freq, cenfreq, l1, real_or_imag):
def plot_cs(cs, csm, freq, cenfreq, l1, real_or_imag):
    """ Plot of cross-spectra (real or imag part only)

    Inputs:
    -------
    cs - np.ndarray(ndim=2, dtype=complex)
        cross-spectra for m >= 0
    csm - np.ndarray(ndim=2, dtype=complex)
        cross-spectra for m <= 0
    freq - np.ndarray(ndim=1, dtype=float)
        array containing frequency
    cenfreq - float
        central frequency value in microHz
    l1 - int
        spherical harmonic degree
    real_or_imag - int
        1 = real; 2 = imag

    Outputs:
    --------
    fig - matplotlib figure
        Figure containing plot of cross-spectra

    """
    plot_data = cs.real if real_or_imag == 1 else cs.imag
    plot_datam = csm.real if real_or_imag == 1 else csm.imag
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    ar = (freq[-1]-freq[0])/l1
    im = plt.imshow(plot_data, vmax=plot_data.max(), aspect=ar,
                    extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0])
    plt.colorbar(im)

    plt.subplot(122)
    im = plt.imshow(plot_datam, vmax=plot_datam.max(), aspect=ar,
                    extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0])
    plt.colorbar(im)
    plt.tight_layout()
    return fig
# }}} plot_cs(cs)


# {{{ def plot_cs_sum(cs, csm, freq, cenfreq, l1):
def plot_cs_sum(cs, csm, freq, cenfreq, l1):
    """ Generates plots of cross-spectra summed over m

    Inputs:
    -------
    cs - np.ndarray(ndim=2, dtype=complex)
        cross-spectra for m >= 0
    csm - np.ndarray(ndim=2, dtype=complex)
        cross-spectra for m <= 0
    freq - np.ndarray(ndim=1, dtype=float)
        array containing frequency
    cenfreq - float
        central frequency value in microHz
    l1 - int
        spherical harmonic degree

    Outputs:
    --------
    fig - matplotlib figure
        Figure containing plot of cross-spectra

    """
    fig = plt.figure()
    plt.subplot(121)
    plt.title("Summation over m >= 0")
    plt.plot(freq-cenfreq, np.sum(cs.real, axis=0)/l1,
             linewidth=0.35, color="blue", label="real")
    plt.plot(freq-cenfreq, np.sum(cs.imag, axis=0)/l1,
             linewidth=0.35, color="red", label="real")
    plt.legend()

    plt.subplot(122)
    plt.title("Summation over m <= 0")
    plt.plot(freq-cenfreq, np.sum(csm.real, axis=0)/l1,
             linewidth=0.35, color="blue", label="real")
    plt.plot(freq-cenfreq, np.sum(csm.imag, axis=0)/l1,
             linewidth=0.35, color="red", label="real")
    plt.legend()
    return fig
# }}} plot_cs_sum(cs, csm, freq, cenfreq, l1)


if __name__ == "__main__":
    # time domain and frequency domain
    t = np.linspace(0, 72*24*3600*daynum, tsLen*daynum)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(t.shape[0], dt)*1e6
    df = freq[1] - freq[0]  # frequencies in microHz

    # reading input values
    l1 = args.l
    n1 = args.n
    l2 = args.lp
    n2 = args.np
    compute_norms = args.norms

    if l2 < l1:
        ltemp = l2
        ntemp = n2
        l2 = l1
        n2 = n1
        l1 = ltemp
        n1 = ntemp

    # reading the leakage matrix
    print("Reading leakage matrix ", end=' '),
    sys.stdout.flush()
    t1 = time.time()
    rleaks = fits.open('/home/g.samarth/leakage/rleaks1.fits')[0].data
    horleaks = fits.open('/home/g.samarth/leakage/horleaks1.fits')[0].data
    t2 = time.time()
    print(f" -- DONE. Time taken = {t2-t1:6.2f} seconds")

    # considering only frequencies in a window
    # where cross spectrum is significant
    cenfreq, cenfwhm, cenamp = cdata.findfreq(data, l1, n1, 0)
    indm = cdata.locatefreq(freq, cenfreq - l1*0.6)
    indp = cdata.locatefreq(freq, cenfreq + l1*0.6)
    freq = freq[indm:indp]

    cs = np.zeros((l1+1, freq.shape[0]), dtype=complex)
    csm = np.zeros((l1+1, freq.shape[0]), dtype=complex)
    phi1sum = np.zeros(freq.shape[0], dtype=complex)

    rleaks1 = rleaks[:, :, l1, :]
    rleaks2 = rleaks[:, :, l2, :]
    horleaks1 = horleaks[:, :, l1, :]
    horleaks2 = horleaks[:, :, l2, :]

    omeganl1, fwhmnl1, amp1 = cdata.findfreq(data, l1, n1, 0)
    omeganl2, fwhmnl2, amp2 = cdata.findfreq(data, l2, n2, 0)

    if compute_norms:
        norm = 1.0
    else:
        # old norms file
        cnl1 = cdata.loadnorms(l1, n1, 1)
        cnl2 = cdata.loadnorms(l2, n2, 1)

        # new norms
        cnl1 = np.loadtxt(writedir + f"norm_{l1:03d}_{n1:02d}")
        cnl2 = np.loadtxt(writedir + f"norm_{l2:03d}_{n2:02d}")

        norm1 = amp1**2 * omeganl1 * omeganl1 * cnl1 * fwhmnl1 * twopiemin6**3
        norm2 = amp2**2 * omeganl2 * omeganl2 * cnl2 * fwhmnl2 * twopiemin6**3
        norm = sqrt(norm1*norm2)

    t1 = time.time()
    for m1 in range(l1+1):
        for dell1 in range(-dl, dl+1):
            dell2 = dell1 - l2 + l1
            l1p = l1 + dell1  # l2p = l1p
            cij = 1.0  # 0.8 + 0.2*np.random.rand()
            for dem in range(-dm, dm+1):
                m1p = m1 + dem
                if (abs(m1p) <= l1p):
                    omeganl1p, fwhmnl1p, amp1p = cdata.findfreq(data,
                                                                l1p, n1, m1p)
                    mix = (274.8*1e2*(l1p+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1p**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m1+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m1+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m1+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m1+249]
                    phi1p = cdata.lorentzian(omeganl1p, fwhmnl1p, freq)
                    phi1sum += leak1*leak2 * (phi1p*phi1p.conjugate()) * cij
                    if abs(phi1sum.imag).sum() > 0:
                        print(f" m1 = {m1}, dem = {dem}, dell1 = {dell1}")
                        exit()
        cs[m1, :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print(f"Time taken for computation (positive m) = " +
          f"{(t2 - t1):6.2f} seconds")

    t1 = time.time()
    for m1 in range(-l1, 1):
        for dell1 in range(-dl, dl+1):
            dell2 = dell1 - l2 + l1
            l1p = l1 + dell1  # l2p = l1p
            cij = 1.0  # 0.7 + 0.3*np.random.rand()
            for dem in range(-dm, dm+1):
                m1p = m1 + dem  # m2p = m1p
                if (abs(m1p) <= l1p):
                    omeganl1p, fwhmnl1p, amp1p = cdata.findfreq(data,
                                                                l1p, n1, m1p)
                    mix = (274.8*1e2*(l1p+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1p**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m1+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m1+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m1+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m1+249]
                    phi1p = cdata.lorentzian(omeganl1p, fwhmnl1p, freq)
                    phi1sum += leak1*leak2 * (phi1p*phi1p.conjugate()) * cij
        csm[abs(m1), :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print(f"Time taken for computation (negative m) = " +
          f"{(t2 - t1):6.2f} seconds")

    # plotting before derotation
#    fig = plot_cs(cs, csm, freq, cenfreq, l1, 1)
#    plt.show()

    # derotating the cross spectra
#    cs = derotate(cs, l1, n1, freq, 1)
#    csm = derotate(csm, l1, n1, freq, -1)

    # plotting after derotation
#    fig = plot_cs(cs, csm, freq, cenfreq, l1, 1)
#    plt.show()

    # writing cross spectra to files
    normname = "norm" if compute_norms else ""
    fp_name = writedir + f"csp_synth_{n1:02d}_{l1:03d}_{l2:03d}" +\
        normname + ".npy"
    fm_name = writedir + f"csm_synth_{n1:02d}_{l1:03d}_{l2:03d}" +\
        normname + ".npy"
    ff_name = writedir + f"freq_synth_{n1:02d}_{l1:03d}_{l2:03d}" +\
        normname + ".npy"
    np.save(fp_name, cs)
    np.save(fm_name, csm)
    np.save(ff_name, freq)

    t2 = time.time()
    print(f"Time taken = {(t2-t1):7.3f} seconds")

    # plotting average over positive and negative m
    if args.plot:
        fig = plot_cs_sum(cs, csm, freq, cenfreq, l1)
        plt.show()

from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
import scipy.ndimage as sciim
from astropy.io import fits
from math import sqrt, pi
import numpy as np
import argparse
import time
import sys

sys.path.append('/home/g.samarth/') # location of heliosPy directory
from heliosPy import datafuncs as cdata

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


def derotate(phi, l, n, freq, pm):
    """Derotate the given cross-spectra
    Inputs:
    -------
    Outputs:
    --------
    """
    data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')
    phinew = np.zeros(phi.shape, dtype=complex)
    for m in range(l+1):
        a1 = finda1(data, l, n, m)
        const = -1*pm*a1*1e-9*72*24*3600
        _real = sciim.shift(phi[m, :].real, const, mode='wrap', order=1)
        _imag = sciim.shift(phi[m, :].imag, const, mode='wrap', order=1)
        phinew[m, :] = _real + 1j*_imag
    return phinew


if __name__=="__main__":
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

    # time domain and frequency domain
    t = np.linspace(0, 72*24*3600*daynum, tsLen*daynum)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(t.shape[0], dt)*1e6
    df = freq[1] - freq[0]  # frequencies in microHz

    # reading mode parameters data
    data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')

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
                    omeganl1, fwhmnl1, amp1 = cdata.findfreq(data, l21, n2, m2)
                    mix = (274.8*1e2*(l21+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m+249]
                    phi1 = cdata.lorentzian(omeganl1, fwhmnl1, freq)
                    phi1sum = phi1sum +\
                        leak1*leak2 * phi1*phi1.conjugate() * tempnorm
        cs[m, :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print("Time taken for computation (positive m) = %6.2f seconds" %(t2 - t1))

    t1 = time.time()
    for m in range(-l1, 1):
        for dell1 in range(-dl, dl+1):
            dell2 = dell1 - l2 + l1
            l21 = l1 + dell1
            tempnorm = 1.0  # 0.7 + 0.3*np.random.rand()
            for dem in range(-dm, dm+1):
                m2 = m+dem
                if (abs(m2)<=l21):
                    omeganl1, fwhmnl1, amp1 = cdata.findfreq(data, l21, n2, m2)
                    mix = (274.8*1e2*(l21+0.5) /
                           ((twopiemin6)**2*rsun))/omeganl1**2
                    leak1 = rleaks1[dem+dm_mat, dell1+dl_mat, m+249] +\
                        mix*horleaks1[dem+dm_mat, dell1+dl_mat, m+249]
                    leak2 = rleaks2[dem+dm_mat, dell2+dl_mat, m+249] +\
                        mix*horleaks2[dem+dm_mat, dell2+dl_mat, m+249]
                    phi1 = cdata.lorentzian(omeganl1, fwhmnl1, freq)
                    phi1sum = phi1sum +\
                        leak1*leak2 * phi1*phi1.conjugate() * tempnorm
        csm[abs(m), :] = phi1sum*norm
        phi1sum = 0.0*phi1sum
    t2 = time.time()
    print("Time taken for computation (negative m) = %6.2f seconds" %(t2 - t1))
    
    # derotating the cross spectra
    cs = derotate(cs, l1, n1, freq, 1) 
    csm = derotate(csm, l1, n1, freq, -1) 
    
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
    print("Time taken = %7.3f seconds" %(t2-t1))
    '''
    # plotting the cross-spectra
    ar = (freq[-1]-freq[0])/l1
    plt.imshow(cs.real, vmax = (cs.real).max(), aspect=ar, extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0]); 
    plt.colorbar(); plt.show(); plt.close();
    
    plt.imshow(csm.real, vmax = (csm.real).max(), aspect=ar, extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0]); 
    plt.colorbar(); plt.show(); plt.close();
    '''

    # plotting average over positive and negative m
    if args.plot:
        plt.plot(freq-cenfreq, np.sum(cs.real, axis=0)/l1, linewidth=0.35); plt.show()
        plt.plot(freq-cenfreq, np.sum(csm.real, axis=0)/l1, linewidth=0.35); plt.show()

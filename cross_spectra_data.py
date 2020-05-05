from numpy.polynomial.legendre import legval
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sciim
from math import sqrt
import numpy as np
import argparse
import time
import sys


# reading input values
parser = argparse.ArgumentParser()
parser.add_argument("--l",
                    help="spherical harmonic degree for mode 1",
                    type=int)
parser.add_argument("--lp",
                    help="spherical harmonic degree for mode 2",
                    type=int)
parser.add_argument("--n", help="radial order for mode 1", type=int)
parser.add_argument("--np", help="radial order for mode 2", type=int)
parser.add_argument("--plot",
                    help="plot results after computation",
                    action="store_true")
args = parser.parse_args()


# importing custom functions
sys.path.append('/home/g.samarth/')
from heliosPy import iofuncs as cio
from heliosPy import datafuncs as cdata


def plot_cs_contour(freq, cs, cenfreq, l1, pm):
    ar = (freq[-1]-freq[0])/l1
    l1 = l1 if pm > 0 else -l1

    fig = plt.figure()
    im = plt.imshow(cs.real, cmap="gray",
                    vmax=(cs.real).max()/3, aspect=ar,
                    extent=[freq[0]-cenfreq, freq[-1]-cenfreq, l1, 0])
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("m")
    cbar_min = cs.min()
    cbar_max = cs.max()/3
    cbar_step = (cbar_max - cbar_min)/8
    cbar = plt.colorbar(im)
    cbar.ax.set_yticklabels(['{:.1e}'.format(x)
                             for x in np.arange(cbar_min,
                                                cbar_max+cbar_step,
                                                cbar_step)],
                            fontsize=12, weight='bold')
    return fig


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
    return totsplit - 31.7*m


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


if __name__ == "__main__":
    # directories
    writedir = '/scratch/g.samarth/csfit/'

    # loading and computation times
    delt_load = 0.0
    delt_compute = 0.0

    # calculation parameters
    daynum = 1              # length of time series
    dayavgnum = 7   # number of time series to be averaged
    tsLen = 138240  # array length of the time series

    # time domain and frequency domain
    t = np.linspace(0, 72*24*3600*daynum, tsLen*daynum)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(t.shape[0], dt)*1e6
    df = freq[1] - freq[0]  # frequencies in microHz

    # loading mode parameters data
    data = np.loadtxt('/home/g.samarth/leakage/hmi.6328.36')

    l1 = args.l
    n1 = args.n
    l2 = args.lp
    n2 = args.np

    # swapping n, l values if l2>l1
    if l2 < l1:
        ltemp = l2
        ntemp = n2
        l2 = l1
        n2 = n1
        l1 = ltemp
        n1 = ntemp

    delta_ell = abs(l2 - l1)

    # considering only frequencies in a window
    # where cross spectrum is significant
    cenfreq, cenfwhm, cenamp = cdata.findfreq(data, l1, n1, 0)
    pmfreq = l1*0.6  # 20
    indm = cdata.locatefreq(freq, cenfreq - pmfreq)
    indp = cdata.locatefreq(freq, cenfreq + pmfreq)
    freq = freq[indm:indp].copy()

    for days in range(dayavgnum):
        day = 6328 + 72*days
        t1 = time.time()

        afftplus1, afftminus1 = cdata.separatefreq(
                cdata.loadHMIdata_avg(l1, day=day))

        if delta_ell == 0:
            afftplus2, afftminus2 = afftplus1, afftminus1
        else:
            afftplus2, afftminus2 = cdata.separatefreq(
                    cdata.loadHMIdata_avg(l2, day=day))

        afft1 = afftplus1[:, indm:indp]
        afft2 = afftplus2[:, indm:indp]
        afft1m = afftminus1[:, indm:indp]
        afft2m = afftminus2[:, indm:indp]
        t2 = time.time()
        delt_load += t2 - t1

        # Calculating the cross-spectrum
        if days == 0:
            t1 = time.time()
            cs = afft1.conjugate()*afft2[:(l1+1), :]
            csm = afft1m.conjugate()*afft2m[:(l1+1), :]
            t2 = time.time()
        else:
            t1 = time.time()
            cs += afft1.conjugate()*afft2[:(l1+1), :]
            csm += afft1m.conjugate()*afft2m[:(l1+1), :]
            t2 = time.time()
        delt_compute += t2 - t1
    print(cs.real.max())
    cs /= dayavgnum
    csm /= dayavgnum
    print(f"Time series load time = {delt_load:7.3f} seconds")
    print(f"Cross spectra computation time = {delt_compute:7.3f} seconds")
    print(f"cenfreq = {cenfreq}")

    '''
    # plotting before derotation
    ar = (freq[-1]-freq[0])/l1
    im = plt.imshow(cs.real, cmap="gray", vmax=(cs.real).max()/5,
                    aspect=ar, extent=[freq[0]-cenfreq, freq[-1]-cenfreq,
                                       l1, 0])
    cbar_min = cs.min()
    cbar_max = cs.max()/3
    cbar_step = (cbar_max - cbar_min)/8
    cbar = plt.colorbar(im)
    cbar.ax.set_yticklabels(['{:.1e}'.format(x)
                             for x in np.arange(cbar_min,
                                                cbar_max+cbar_step,
                                                cbar_step)],
                            fontsize=12, weight='bold')
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("m")
    plt.show()
    plt.close()

    plt.imshow(csm.real, cmap="gray",
               vmax=(csm.real).max()/5, aspect=ar,
               extent=[freq[0]-cenfreq, freq[-1]-cenfreq, -l1, 0])
    plt.colorbar()
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("m")
    plt.show()
    plt.close()

    fig = plot_cs_contour(freq, cs, cenfreq, l1, 1)
    plt.show()
    plt.close()

    fig = plot_cs_contour(freq, csm, cenfreq, l1, -1)
    plt.show()
    plt.close()
    '''

    # derotating the cross spectra
    cs = derotate(cs, l1, n1, freq, 1)
    csm = derotate(csm, l1, n1, freq, -1)

    # restricting frequencies to within 25 \micro Hz of cenfreq
    pmfreq = 35
    indm = cdata.locatefreq(freq, cenfreq - pmfreq)
    indp = cdata.locatefreq(freq, cenfreq + pmfreq)
    freq = freq[indm:indp].copy()
    cs = cs[:, indm:indp].copy()
    csm = csm[:, indm:indp].copy()

#    fig = plot_cs_contour(freq, cs, cenfreq, l1, 1)
#    plt.show()
#    fig = plot_cs_contour(freq, csm, cenfreq, l1, -1)
#    plt.show()


    # plotting average over positive and negative m
    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(freq-cenfreq, np.sum(cs.real, axis=0)/l1,
                 linewidth=0.35, color='blue', label='+m')

        plt.subplot(122)
        plt.plot(freq-cenfreq, np.sum(csm.real, axis=0)/l1,
                 linewidth=0.35, color='red', label='-m')
        plt.tight_layout()
        plt.show()

    fp_name = writedir + f"csp_data_{n1:02d}_{l1:03d}_{l2:03d}.npy"
    fm_name = writedir + f"csm_data_{n1:02d}_{l1:03d}_{l2:03d}.npy"
    ff_name = writedir + f"freq_data_{n1:02d}_{l1:03d}_{l2:03d}.npy"
    np.save(fp_name, cs)
    np.save(fm_name, csm)
    np.save(ff_name, freq)

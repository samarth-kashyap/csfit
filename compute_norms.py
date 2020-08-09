from heliosPy import datafuncs as cdata
from globalvars import globalvars
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import argparse
import os


# {{{ def fremove(fname):
def fremove(fname):
    try:
        os.system("rm "+fname)
        print("Removing temp file "+fname)
    except OSError:
        print("Can't remove "+fname)
    return 0
# }}} fremove(fname)


# {{{ ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("--force",
                    help="Force computes norms",
                    action="store_true")
parser.add_argument("--l", help="spherical harmonic degree",
                    type=int)
parser.add_argument("--n", help="radial order",
                    type=int)
args = parser.parse_args()
# }}} parser


def plot_cs_sum(cs_data_sum, cs_synth_sum, dcshift, norm, freq_synth,
                freq_data, cenfreq, csm_data_sum, csm_synth_sum, dcmshift):
    fig = plt.figure(figsize=(9, 5))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size='14')
    plt.subplot(122)
    plt.xlim(-9, 23)
    plt.ylim(0, 2*dcshift +
             max(abs(cs_data_sum).max(), abs(cs_synth_sum).max()*norm))
    plt.plot(freq_synth -
             cenfreq, dcshift + cs_synth_sum*norm, 'r', label='synthetic')
    plt.plot(freq_data - cenfreq, dcshift + cs_data_sum, 'b', label='data')
    plt.title("$\sum_{m \ge 0} |\phi^{200, m}|^2$")
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("amplitude")
    plt.legend()

    plt.subplot(121)
    plt.xlim(-9, 23)
    plt.ylim(0, 2*dcshift +
             max(abs(csm_data_sum).max(), abs(csm_synth_sum).max()*norm))
    plt.plot(freq_synth -
             cenfreq, dcmshift + csm_synth_sum*norm, 'r', label='synthetic')
    plt.plot(freq_data - cenfreq, dcshift + csm_data_sum, 'b', label='data')
    plt.title("$\sum_{m < 0} |\phi^{200, m}|^2$")
    plt.xlabel("$\omega - \omega_{nl}$ in $\mu$Hz")
    plt.ylabel("amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/g.samarth/ps200.png")
    return fig


gvar = globalvars()
l1 = int(args.l)
n1 = int(args.n)

writedir = gvar.write_dir
normname = f"norm_{l1:03d}_{n1:02d}"

data = np.loadtxt(gvar.leak_dir + 'hmi.6328.36')

if args.force:
    normsexist = False
else:
    try:
        a = np.loadtxt(writedir + normname)
        normsexist = True if abs(a) > 0 else False
    except OSError:
        normsexist = False

print(f" normsexist = {normsexist}")
if not normsexist:
    print(f" normsexist = {normsexist}")
    print(f"Computing data")
    os.system(f"python {gvar.prog_dir}cross_spectra_data.py --l " +
              f"{l1} --n {n1} --lp {l1} --np {n1}")
    print("Computing synth")
    os.system(f"python {gvar.prog_dir}cross_spectra_synth.py --l " +
              f"{l1} --n {n1} --lp {l1} --np {n1} --norms")
else:
    print("Norms exist, reading from " + writedir + normname)


normname2 = "" if normsexist else "norm"
fp_name_s = writedir + f"csp_synth_{n1:02d}_{l1:03d}_{l1:03d}" +\
    normname2 + ".npy"
fm_name_s = writedir + f"csm_synth_{n1:02d}_{l1:03d}_{l1:03d}" +\
    normname2 + ".npy"
ff_name_s = writedir + f"freq_synth_{n1:02d}_{l1:03d}_{l1:03d}" +\
    normname2 + ".npy"
fp_name_d = writedir + f"csp_data_{n1:02d}_{l1:03d}_{l1:03d}.npy"
fm_name_d = writedir + f"csm_data_{n1:02d}_{l1:03d}_{l1:03d}.npy"
ff_name_d = writedir + f"freq_data_{n1:02d}_{l1:03d}_{l1:03d}.npy"

cs_synth = np.load(fp_name_s)
csm_synth = np.load(fm_name_s)
cs_data = np.load(fp_name_d)
csm_data = np.load(fm_name_d)
freq_synth = np.load(ff_name_s)
freq_data = np.load(ff_name_d)

cs_synth_sum = cs_synth.sum(axis=0)
csm_synth_sum = csm_synth.sum(axis=0)
cs_data_sum = cs_data.sum(axis=0)
csm_data_sum = csm_data.sum(axis=0)

cenfreq, fwhm, amp = cdata.findfreq(data, l1, n1, 0)

calc_dc_shift = True

if calc_dc_shift:
    indm_dc = cdata.locatefreq(freq_data, cenfreq - 40)
    indp_dc = cdata.locatefreq(freq_data, cenfreq - 20)
    dcshift = cs_data_sum[indm_dc:indp_dc].mean()
    dcmshift = csm_data_sum[indm_dc:indp_dc].mean()
    cs_data_sum -= dcshift
    csm_data_sum -= dcmshift
else:
    dcshift = 0

indp_data = cdata.locatefreq(freq_data, cenfreq + 2*fwhm)
indp_synth = cdata.locatefreq(freq_synth, cenfreq + 2*fwhm)
indm_data = cdata.locatefreq(freq_data, cenfreq - 2*fwhm)
indm_synth = cdata.locatefreq(freq_synth, cenfreq - 2*fwhm)

# norm definition 1 -- scaling factor by equating areas around 1 linewidth
dr = cs_synth_sum[indm_synth:indp_synth].sum() +\
    csm_synth_sum[indm_synth:indp_synth].sum()
nr = cs_data_sum[indm_data:indp_data].sum() +\
    csm_data_sum[indm_data:indp_data].sum()

'''
# norm definition 2 -- ratio of peaks at cenfreq
dr = (cs_synth_sum[indm_synth:indp_synth] +
      csm_synth_sum[indm_synth:indp_synth]).max()
nr = (cs_data_sum[indm_data:indp_data] +
      csm_data_sum[indm_data:indp_data]).max()

# norm definition 2b -- ratio of peaks at cenfreq (positive m)
dr = cs_synth_sum[indm_synth:indp_synth].max()
nr = cs_data_sum[indm_data:indp_data].max()

# norm definition 4 -- scaling factor using least square minimization
indp_data = cdata.locatefreq(freq_data, cenfreq + 70*fwhm)
indp_synth = cdata.locatefreq(freq_synth, cenfreq + 70*fwhm)
indm_data = cdata.locatefreq(freq_data, cenfreq - 70*fwhm)
indm_synth = cdata.locatefreq(freq_synth, cenfreq - 70*fwhm)
cs_synth_sum4 = cs_synth_sum[indm_synth:indp_synth]
cs_data_sum4 = cs_data_sum[indm_data:indp_data]
nr = (cs_synth_sum4 * cs_data_sum4).sum()
dr = (cs_synth_sum4**2).sum()
'''
norm = nr/dr
norm *= 1.1
print(dcshift)
twopiemin6 = 2*pi*1e-6
cnl = norm/amp**2/cenfreq**2/fwhm/twopiemin6**3

fig = plot_cs_sum(cs_data_sum, cs_synth_sum, dcshift, norm, freq_synth,
                  freq_data, cenfreq, csm_data_sum, csm_synth_sum, dcmshift)
plt.show(fig)
"""
if not normsexist:
    ax = plt.subplot(111)
    plt.xlim(-25, 25)
    plt.ylim(0, 2*dcshift +
             max(abs(cs_data_sum).max(), abs(cs_synth_sum).max()*norm))
    ax.plot(-freq_synth +
            cenfreq, dcshift + cs_synth_sum*norm, 'r', label='synthetic')
    ax.plot(freq_data - cenfreq, dcshift + cs_data_sum, 'b', label='data')
    ax.legend()
    plt.show()
    np.savetxt(writedir + normname, np.array([abs(cnl)]), fmt='%.18e')
    print("Writing " + writedir + normname)
else:
    ax = plt.subplot(111)
    plt.xlim(-25, 25)
    plt.ylim(0, 2*dcshift +
             max(abs(cs_data_sum).max(), abs(cs_synth_sum).max()*norm))
    ax.plot(-freq_synth +
            cenfreq, dcshift + cs_synth_sum*norm, 'r', label='synthetic')
    ax.plot(freq_data - cenfreq, dcshift + cs_data_sum, 'b', label='data')
    ax.legend()
    plt.show()
"""
print(f" cnl = {cnl}")

if not normsexist:
    fremove(fp_name_s)
    fremove(fm_name_s)
    fremove(ff_name_s)

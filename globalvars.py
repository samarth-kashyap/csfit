from math import pi
import os


class globalvars():
    """Class to store all the global variables

    """
    def __init__(self):
        # path of the leakage matrices and freq splitting data
        self.leak_dir = "/scratch/g.samarth/HMIDATA/leakmat/"

        # path where output files will be written
        self.write_dir = "/scratch/g.samarth/csfit/"

        # path where program is located
        self.prog_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

        self.dl = 6
        self.dm = 15
        self.dl_mat = 6
        self.dm_mat = 15
        self.rsun = 6.9598e10
        self.twopiemin6 = 2*pi*1e-6

        self.daynum = 1      # length of time series
        self.tsLen = 138240  # array length of the time series

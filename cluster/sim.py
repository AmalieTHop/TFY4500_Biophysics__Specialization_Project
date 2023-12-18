import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch

import glob

# sims
import models.IVIMNET.IVIMNET.deep as from_deep   #import IVIMNET.deep as deep
import models.IVIMNET.IVIMNET.simulations as from_sim   #import IVIMNET.simulations as sim
from models.IVIMNET.hyperparams import hyperparams as from_hps


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr


def run_simulation(snr):
    print(f"SNR: {snr}")
    # load hyperparameter
    arg = from_hps()
    arg = from_deep.checkarg(arg)

    # simulate
    from_sim.sim(snr, arg)

    # done
    print("Done!") 


run_simulation(snr)

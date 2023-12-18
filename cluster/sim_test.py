"""
September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# import
import numpy as np
import models.IVIMNET.IVIMNET.simulations as sim
import models.IVIMNET.IVIMNET.deep as deep
from models.IVIMNET.hyperparams import hyperparams as hp_example

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr


def run_sims(snr):
    print(f"SNR: {snr}")

    # load hyperparameter
    arg = hp_example()
    arg = deep.checkarg(arg)

    dst_dir = 'simulations/simulations_5bvalues_rep25_0311code'
    sim.sim(snr, dst_dir, arg)

run_sims(snr)

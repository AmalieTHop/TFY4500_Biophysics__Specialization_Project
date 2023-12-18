#!/bin/sh
module purge
module load Anaconda3/2020.07

sbatch --partition=CPUQ --account=nv-fys --time=06:00:00 --nodes=5 --ntasks-per-node=20 --mem=25000 --job-name="ALL" --output=patients/all.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python patients_combined.py"

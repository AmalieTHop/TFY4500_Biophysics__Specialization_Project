#!/bin/sh
module purge
module load Anaconda3/2020.07

sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr10_rep25" --output=simulations/snr10_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 10"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr15_rep25" --output=simulations/snr15_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 15"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr20_rep25" --output=simulations/snr20_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr25_rep25" --output=simulations/snr25_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr30_rep25" --output=simulations/snr30_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 30"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr50_rep25" --output=simulations/snr50_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 50"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr75_rep25" --output=simulations/snr75_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 75"
sbatch --partition=CPUQ --account=nv-fys --time=03:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr100_rep25" --output=simulations/snr100_rep25.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim.py --snr 100"
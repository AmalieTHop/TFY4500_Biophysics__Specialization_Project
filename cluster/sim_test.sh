#!/bin/sh
module purge
module load Anaconda3/2020.07

sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr8_r25_b5" --output=simulations/snr8_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 8"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr10_r25_b5" --output=simulations/snr10_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 10"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr12_r25_b5" --output=simulations/snr12_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 12"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr15_r25_b5" --output=simulations/snr15_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 15"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr20_r25_b5" --output=simulations/snr20_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 20"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr25_r25_b5" --output=simulations/snr25_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 25"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr33_r25_b5" --output=simulations/snr33_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 33"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr50_r25_b5" --output=simulations/snr50_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 50"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr75_r25_b5" --output=simulations/snr75_rep25_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 75"
sbatch --partition=CPUQ --account=nv-fys --time=05:30:00 --nodes=1 --ntasks-per-node=4 --mem=12000 --job-name="snr100_r25_b5" --output=simulations/snr100_re525_5bvals.out --mail-user=amalieth@stud.ntnu.no --mail-type=ALL --wrap="python sim_test.py --snr 100"
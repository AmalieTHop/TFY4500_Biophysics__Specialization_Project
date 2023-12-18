import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch

import glob

# patients
import models.run_models as from_models


def run_combined_patient_data():
    patient_ids = ['EMIN_1001', 'EMIN_1003', 'EMIN_1005', 'EMIN_1007', 'EMIN_1008',
                   'EMIN_1011', 'EMIN_1016', 'EMIN_1019', 'EMIN_1020', 'EMIN_1022',
                   'EMIN_1023', 'EMIN_1031', 'EMIN_1032', 'EMIN_1038', #'EMIN_1041',
                   'EMIN_1042', 'EMIN_1044', 'EMIN_1045', 'EMIN_1048', #'EMIN_1050',
                   'EMIN_1051', 'EMIN_1055', 'EMIN_1057', 'EMIN_1058', 'EMIN_1060',
                   'EMIN_1061', 'EMIN_1064', 'EMIN_1066', 'EMIN_1068', 'EMIN_1074',
                   'EMIN_1075', 'EMIN_1077', 'EMIN_1078', 'EMIN_1079', 'EMIN_1080',
                   'EMIN_1081', 'EMIN_1083', 'EMIN_1084', 'EMIN_1086', 'EMIN_1090',
                   'EMIN_1092', 'EMIN_1093', 'EMIN_1096', 'EMIN_1097', 'EMIN_1099']
    ivim_net = from_models.IVIM_NN_combined_training(patient_ids)
    #trained_model_ivim_net = ivim_net.train_ivim()
    trained_model_ivim_net = torch.load(f'data/results/trained_model_net')
    trained_model_ivim_net.eval()
    ivim_net.predict_ivim(trained_model_ivim_net)
    ivim_net.calculate_descriptive_statistics()
    
run_combined_patient_data()
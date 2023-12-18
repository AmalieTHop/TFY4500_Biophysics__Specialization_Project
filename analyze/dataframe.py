
import numpy as np
import pandas as pd

import glob
import os



def create_df_model_params(patient_ids):
    GTV_root_names = ['GTVp', 'GTVn']

    df_tot = pd.DataFrame(columns=['patient_id', 'GTV_name', 'ACD', 
                               'Dt_conv', 'Fp_conv', 'Dp_conv',
                               'Dt_seg', 'Fp_seg', 'Dp_seg',
                               'Dt_net', 'Fp_net', 'Dp_net',
                               'HPV', 'T', 'N'
                               ])
    
    df_endpts = pd.read_excel(f'analyze/Patient_outcomes.xlsx')
    

    for patient_id in patient_ids:
        df_patient_endpt = df_endpts.query(f"patient_id == '{patient_id}'")

        for GTV_root_name in GTV_root_names:
            
            full_GTV_root_names = glob.glob(f'data/results/{patient_id}/{GTV_root_name}*')
            for full_GTV_root_name in full_GTV_root_names:
                desc_stats_adc = np.loadtxt(os.path.join(full_GTV_root_name, 'ADC/descriptive_statistics.txt'))
                desc_stats_conv = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/conv/descriptive_statistics.txt'))
                desc_stats_seg = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/seg/descriptive_statistics.txt'))
                desc_stats_net = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/net/descriptive_statistics.txt'))

                new_row = pd.DataFrame({'patient_id': patient_id, 
                            'GTV_name': GTV_root_name, 
                            'ACD': [desc_stats_adc[1]], 
                            'Dt_conv': [desc_stats_conv[0][1]], 
                            'Fp_conv': [desc_stats_conv[1][1]], 
                            'Dp_conv': [desc_stats_conv[2][1]],
                            'Dt_seg': [desc_stats_seg[0][1]], 
                            'Fp_seg': [desc_stats_seg[1][1]],
                            'Dp_seg': [desc_stats_seg[2][1]],
                            'Dt_net': [desc_stats_net[0][1]],
                            'Fp_net': [desc_stats_net[1][1]],
                            'Dp_net': [desc_stats_net[2][1]],
                            'HPV': str(df_patient_endpt['HPV'].item()),
                            'T': str(df_patient_endpt['T'].item()), 
                            'N': str(df_patient_endpt['N'].item()),
                            })
                
                df_tot = pd.concat([df_tot, new_row])
    
    df_tot.to_csv('analyze/df_tot.csv', index=False)
    
    return df_tot

"""
def create_df_patient_stats(patient_ids):
    GTV_root_names = ['GTVp', 'GTVn']

    df_patient_stats = pd.DataFrame(columns=['patient_id', 'GTV_name', 'gender', 'age', 'T', 'N'])
    df_endpts = pd.read_excel(f'analyze/Patient_outcomes.xlsx')

    for patient_id in patient_ids:
        df_patient_endpt = df_endpts.query(f"patient_id == '{patient_id}'")
"""
    
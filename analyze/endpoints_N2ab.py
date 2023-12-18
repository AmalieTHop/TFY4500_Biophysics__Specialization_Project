
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns

import glob
import os


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat='%1.1f', offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat



def create_df_model_params(patient_ids):
    GTV_root_names = ['GTVp', 'GTVn']

    df_tot = pd.DataFrame(columns=['patient_id', 'GTV_name', 'ADC', 
                               'Dt_conv', 'Fp_conv', 'Dp_conv',
                               'Dt_seg', 'Fp_seg', 'Dp_seg',
                               'Dt_net', 'Fp_net', 'Dp_net',
                               'HPV', 'T', 'N'
                               ])
    
    df_endpts = pd.read_excel(f'data/analysis/Patient_outcomes.xlsx')
    

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
                            'ADC': [desc_stats_adc[1]], 
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
    
    df_tot.to_csv(f'data/analysis/df_endpoints.csv', index=False)
    
    return df_tot



class EndpointAnalysis:
    def __init__(self):
        self.dst_dir = f'data/plots'
        self.df_tot = pd.read_csv(f'data/analysis/df_endpoints.csv')
        

    def boxplot_ivim_t_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        #fig.suptitle(f'T in TNM', fontsize=25)

        ax00 = sns.boxplot(ax=axes[0, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax01 = sns.boxplot(ax=axes[0, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax02 = sns.boxplot(ax=axes[0, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax00)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax01)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax02)

        ax10 = sns.boxplot(ax=axes[1, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax11 = sns.boxplot(ax=axes[1, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax12 = sns.boxplot(ax=axes[1, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax10)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax11)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax12)

        ax20 = sns.boxplot(ax=axes[2, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax21 = sns.boxplot(ax=axes[2, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax22 = sns.boxplot(ax=axes[2, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_conv", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax20)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_seg", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax21)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_net", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax22)


        ax00.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax01.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='SEG')
        ax02.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='DNN')
        ax10.set(xlabel=None, ylabel = r'$f_p$ [%]', title='LSQ')
        ax11.set(xlabel=None, ylabel = r'$f_p$ [%]', title='SEG')
        ax12.set(xlabel=None, ylabel = r'$f_p$ [%]', title='DNN')
        ax20.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='SEG')
        ax22.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='DNN')

        ax00.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax20.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        ax00.legend(title=None)
        ax01.legend(title=None)
        ax02.legend(title=None)
        ax10.legend(title=None)
        ax11.legend(title=None)
        ax12.legend(title=None)
        ax20.legend(title=None)
        ax21.legend(title=None)
        ax22.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_ivim_T.pdf'))


    def boxplot_adc_t_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.boxplot(ax=ax, data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="ADC", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="ADC", hue="HPV", order=['T1', 'T2', 'T3', 'T4a'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)

        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_adc_T.pdf'))


    def boxplot_ivim_n_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        #fig.suptitle(f'N in TNM', fontsize=25)

        ax00 = sns.boxplot(ax=axes[0, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax01 = sns.boxplot(ax=axes[0, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax02 = sns.boxplot(ax=axes[0, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax00)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax01)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax02)

        ax10 = sns.boxplot(ax=axes[1, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax11 = sns.boxplot(ax=axes[1, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax12 = sns.boxplot(ax=axes[1, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax10)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax11)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax12)

        ax20 = sns.boxplot(ax=axes[2, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax21 = sns.boxplot(ax=axes[2, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax22 = sns.boxplot(ax=axes[2, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_conv", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax20)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_seg", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax21)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_net", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax22)

        ax00.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax01.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='SEG')
        ax02.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='DNN')
        ax10.set(xlabel=None, ylabel = r'$f_p$ [%]', title='LSQ')
        ax11.set(xlabel=None, ylabel = r'$f_p$ [%]', title='SEG')
        ax12.set(xlabel=None, ylabel = r'$f_p$ [%]', title='DNN')
        ax20.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='SEG')
        ax22.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='DNN')

        ax00.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax20.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        ax00.legend(title=None)
        ax01.legend(title=None)
        ax02.legend(title=None)
        ax10.legend(title=None)
        ax11.legend(title=None)
        ax12.legend(title=None)
        ax20.legend(title=None)
        ax21.legend(title=None)
        ax22.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_ivim_N.pdf'))



    def boxplot_adc_n_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.boxplot(ax=ax, data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="ADC", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="ADC", hue="HPV", order=['N0', 'N1', 'N2a', 'N2b'], dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)

        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_adc_N.pdf'))



    def boxplot_ivim_hpv(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        #fig.suptitle(f'T in TNM', fontsize=25)

        ax00 = sns.boxplot(ax=axes[0, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_conv", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax01 = sns.boxplot(ax=axes[0, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_seg", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax02 = sns.boxplot(ax=axes[0, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_net", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_conv", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax00)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_seg", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax01)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dt_net", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax02)

        ax10 = sns.boxplot(ax=axes[1, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_conv", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax11 = sns.boxplot(ax=axes[1, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_seg", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax12 = sns.boxplot(ax=axes[1, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_net", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_conv", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax10)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_seg", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax11)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Fp_net", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax12)

        ax20 = sns.boxplot(ax=axes[2, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_conv", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax21 = sns.boxplot(ax=axes[2, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_seg", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        ax22 = sns.boxplot(ax=axes[2, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_net", showfliers=showfliers, boxprops=boxprops, showmeans=True)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_conv", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax20)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_seg", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax21)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="Dp_net", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax22)


        ax00.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax01.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='SEG')
        ax02.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='DNN')
        ax10.set(xlabel=None, ylabel = r'$f_p$ [%]', title='LSQ')
        ax11.set(xlabel=None, ylabel = r'$f_p$ [%]', title='SEG')
        ax12.set(xlabel=None, ylabel = r'$f_p$ [%]', title='DNN')
        ax20.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='SEG')
        ax22.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='DNN')

        ax00.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax20.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        ax00.legend(title=None)
        ax01.legend(title=None)
        ax02.legend(title=None)
        ax10.legend(title=None)
        ax11.legend(title=None)
        ax12.legend(title=None)
        ax20.legend(title=None)
        ax21.legend(title=None)
        ax22.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_ivim_HPV.pdf'))



    def boxplot_adc_hpv(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.boxplot(ax=ax, data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="ADC",  showfliers=showfliers, boxprops=boxprops, showmeans=True)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="HPV", y="ADC", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)

        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_adc_HPV.pdf'))




    def boxplot_ivim_t_only_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9
        color_strip = 'C0'

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        #fig.suptitle(f'T in TNM', fontsize=25)

        ax00 = sns.boxplot(ax=axes[0, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_conv", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax01 = sns.boxplot(ax=axes[0, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_seg", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax02 = sns.boxplot(ax=axes[0, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_net",  order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_conv", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax00)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_seg", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax01)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dt_net", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax02)

        ax10 = sns.boxplot(ax=axes[1, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_conv", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax11 = sns.boxplot(ax=axes[1, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_seg", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax12 = sns.boxplot(ax=axes[1, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_net", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_conv", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax10)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_seg", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax11)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Fp_net", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax12)

        ax20 = sns.boxplot(ax=axes[2, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_conv", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax21 = sns.boxplot(ax=axes[2, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_seg", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        ax22 = sns.boxplot(ax=axes[2, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_net", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_conv", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax20)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_seg", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax21)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="Dp_net", order=['T1', 'T2', 'T3', 'T4a'], color = color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax22)


        ax00.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax01.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='SEG')
        ax02.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='DNN')
        ax10.set(xlabel=None, ylabel = r'$f_p$ [%]', title='LSQ')
        ax11.set(xlabel=None, ylabel = r'$f_p$ [%]', title='SEG')
        ax12.set(xlabel=None, ylabel = r'$f_p$ [%]', title='DNN')
        ax20.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='SEG')
        ax22.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='DNN')

        ax00.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax20.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_ivim_T_only.pdf'))


    def boxplot_adc_t_only_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9
        color_strip = 'C0'


        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.boxplot(ax=ax, data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="ADC", order=['T1', 'T2', 'T3', 'T4a'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="T", y="ADC", order=['T1', 'T2', 'T3', 'T4a'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)

        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_adc_T_only.pdf'))




    def boxplot_ivim_n_only_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9
        color_strip = 'C0'


        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        #fig.suptitle(f'N in TNM', fontsize=25)

        ax00 = sns.boxplot(ax=axes[0, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_conv", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax01 = sns.boxplot(ax=axes[0, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_seg", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax02 = sns.boxplot(ax=axes[0, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_net", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_conv", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax00)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_seg", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax01)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dt_net", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax02)

        ax10 = sns.boxplot(ax=axes[1, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_conv", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax11 = sns.boxplot(ax=axes[1, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_seg", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax12 = sns.boxplot(ax=axes[1, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_net", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_conv", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax10)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_seg", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax11)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Fp_net", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax12)

        ax20 = sns.boxplot(ax=axes[2, 0], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_conv", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax21 = sns.boxplot(ax=axes[2, 1], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_seg", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        ax22 = sns.boxplot(ax=axes[2, 2], data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_net", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_conv", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax20)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_seg", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax21)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="Dp_net", order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax22)

        ax00.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax01.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='SEG')
        ax02.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]', title='DNN')
        ax10.set(xlabel=None, ylabel = r'$f_p$ [%]', title='LSQ')
        ax11.set(xlabel=None, ylabel = r'$f_p$ [%]', title='SEG')
        ax12.set(xlabel=None, ylabel = r'$f_p$ [%]', title='DNN')
        ax20.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='SEG')
        ax22.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]', title='DNN')

        ax00.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax20.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_ivim_N_only.pdf'))



    def boxplot_adc_n_only_in_tnm(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9
        color_strip = 'C0'


        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.boxplot(ax=ax, data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="ADC", order=['N0', 'N1', 'N2a', 'N2b'], showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_tot.query(f"GTV_name == 'GTVp'"), x="N", y="ADC",  order=['N0', 'N1', 'N2a', 'N2b'], color=color_strip, dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)

        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'endpoints_adc_N_only.pdf'))


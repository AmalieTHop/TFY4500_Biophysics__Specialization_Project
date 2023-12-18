
import numpy as np
import pandas as pd
import pydicom
from scipy.stats import mannwhitneyu
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker
sns.set()

import pickle
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
        #if self._useMathText:
        #    self.format = r'$\mathdefault{%s}$' % self.format



def prepare_boxplot_df(patient_ids, GTV_root_names, model, method = None):
    analysis_dir = f'data/analysis'
    if not os.path.exists(analysis_dir):
                os.makedirs(analysis_dir)

    if model == 'ADC':
        df_tot = pd.DataFrame(columns=['patient_id', 'GTV_name', 'ADC'])
        for patient_id in patient_ids:
            for GTV_root_name in GTV_root_names:
                full_GTV_root_names = glob.glob(f'data/results/{patient_id}/{GTV_root_name}*')
                for full_GTV_root_name in full_GTV_root_names:
                    desc_stats_adc = np.loadtxt(os.path.join(full_GTV_root_name, 'ADC/descriptive_statistics.txt'))
                    new_row =  pd.DataFrame({'patient_id': patient_id, 'GTV_name': GTV_root_name, 'ADC': [desc_stats_adc[1]]})
                    df_tot = pd.concat([df_tot, new_row])

        df_tot.to_csv('data/analysis/df_adc.csv', index=False)
        
    else:
        df_Dt = pd.DataFrame(columns=['patient_id', 'GTV_name', 'method', 'Dt'])
        df_Fp = pd.DataFrame(columns=['patient_id', 'GTV_name', 'method', 'Fp'])
        df_Dp = pd.DataFrame(columns=['patient_id', 'GTV_name', 'method', 'Dp'])
        for patient_id in patient_ids:
            for GTV_root_name in GTV_root_names:
                full_GTV_root_names = glob.glob(f'data/results/{patient_id}/{GTV_root_name}*')
                for full_GTV_root_name in full_GTV_root_names:
                    desc_stats_lsq = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/conv/descriptive_statistics.txt'))
                    desc_stats_seg = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/seg/descriptive_statistics.txt'))
                    desc_stats_dnn = np.loadtxt(os.path.join(full_GTV_root_name, 'IVIM/net/descriptive_statistics.txt'))

                    new_row_Dt_lsq = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'LSQ', 'Dt': [desc_stats_lsq[0][1]]})
                    new_row_Dt_seg = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'SEG', 'Dt': [desc_stats_seg[0][1]]})
                    new_row_Dt_dnn = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'DNN', 'Dt': [desc_stats_dnn[0][1]]})

                    new_row_Fp_lsq = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'LSQ', 'Fp': [desc_stats_lsq[1][1]*100]}) ###
                    new_row_Fp_seg = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'SEG', 'Fp': [desc_stats_seg[1][1]*100]}) ###
                    new_row_Fp_dnn = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'DNN', 'Fp': [desc_stats_dnn[1][1]*100]}) ###

                    new_row_Dp_lsq = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'LSQ', 'Dp': [desc_stats_lsq[2][1]]})
                    new_row_Dp_seg = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'SEG', 'Dp': [desc_stats_seg[2][1]]})
                    new_row_Dp_dnn = pd.DataFrame({'patient_id':patient_id, 'GTV_name': GTV_root_name, 'method':'DNN', 'Dp': [desc_stats_dnn[2][1]]})
                                                  
                    df_Dt = pd.concat([df_Dt, new_row_Dt_lsq, new_row_Dt_seg, new_row_Dt_dnn])
                    df_Fp = pd.concat([df_Fp, new_row_Fp_lsq, new_row_Fp_seg, new_row_Fp_dnn])
                    df_Dp = pd.concat([df_Dp, new_row_Dp_lsq, new_row_Dp_seg, new_row_Dp_dnn])
        
        df_Dt.to_csv('data/analysis/df_ivim_Dt.csv', index=False)
        df_Fp.to_csv('data/analysis/df_ivim_Fp.csv', index=False)
        df_Dp.to_csv('data/analysis/df_ivim_Dp.csv', index=False)


class Boxplot:
    def __init__(self, GTV_root_names, patient_ids = []):

        self.dst_dir = f'data/plots'
        if not os.path.exists(self.dst_dir):
                os.makedirs(self.dst_dir)

        prepare_boxplot_df(patient_ids, GTV_root_names, model='ADC')
        prepare_boxplot_df(patient_ids, GTV_root_names, model='IVIM')

        self.df_adc = pd.read_csv(f'data/analysis/df_adc.csv')
        self.df_Dt = pd.read_csv(f'data/analysis/df_ivim_Dt.csv')
        self.df_Fp = pd.read_csv(f'data/analysis/df_ivim_Fp.csv')
        self.df_Dp = pd.read_csv(f'data/analysis/df_ivim_Dp.csv')



    def plot_ivim_boxplot(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        #fig.suptitle(f'IVIM models', fontsize=25)

        ax0 = sns.boxplot(ax=axes[0], data=self.df_Dt, x='method', y='Dt', hue='GTV_name', showfliers=showfliers, boxprops=boxprops)
        ax1 = sns.boxplot(ax=axes[1], data=self.df_Fp, x='method', y='Fp', hue='GTV_name', showfliers=showfliers, boxprops=boxprops)
        ax2 = sns.boxplot(ax=axes[2], data=self.df_Dp, x='method', y='Dp', hue='GTV_name', showfliers=showfliers, boxprops=boxprops)

        sns.stripplot(data=self.df_Dt, x="method", y="Dt", hue="GTV_name", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax0)
        sns.stripplot(data=self.df_Fp, x="method", y="Fp", hue="GTV_name", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax1)
        sns.stripplot(data=self.df_Dp, x="method", y="Dp", hue="GTV_name", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax2)

        ax0.set(xlabel=None, ylabel = r'$D_t$ [mm$^2$/s]')
        ax1.set(xlabel=None, ylabel = r'$f_p$ [%]')
        ax2.set(xlabel=None, ylabel = r'$D_p$ [mm$^2$/s]')

        ax0.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        
        ax0.legend(title=None)
        ax1.legend(title=None)
        ax2.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'IVIM_collective.pdf'))



    def plot_adc_boxplot(self):
        sns.set()
        params = {'axes.labelsize': 15,
                  'axes.titlesize': 20,
                  'xtick.labelsize': 12.5,
                  'ytick.labelsize': 12.5,
                  'legend.fontsize': 10,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(params)

        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9


        fig, ax = plt.subplots(figsize=(5, 5))
        #fig.suptitle(f'IVIM models', fontsize=25)
 
        ax = sns.boxplot(data=self.df_adc, y='ADC', hue='GTV_name', showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_adc, y="ADC", hue="GTV_name", dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)
        ax.set(xlabel=None, ylabel = r'$ADC$ [mm$^2$/s]')
        ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        ax.legend(title=None)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'ADC_collective.pdf'))



    def mann_whitney_u_test(self):
        adc_gtvp = self.df_adc[self.df_adc['GTV_name'] == 'GTVp']['ADC'].to_numpy()
        adc_gtvn = self.df_adc[self.df_adc['GTV_name'] == 'GTVn']['ADC'].to_numpy()
        U, p_adc_gtv = mannwhitneyu(adc_gtvn, adc_gtvp)
        
        Dt_lsq_gtvp = self.df_Dt[(self.df_Dt['method'] == 'LSQ') & (self.df_Dt['GTV_name'] == 'GTVp')]['Dt'].to_numpy()
        Dt_lsq_gtvn = self.df_Dt[(self.df_Dt['method'] == 'LSQ') & (self.df_Dt['GTV_name'] == 'GTVn')]['Dt'].to_numpy()
        U, p_Dt_conv_gtv = mannwhitneyu(Dt_lsq_gtvp, Dt_lsq_gtvn)

        Dt_seg_gtvp = self.df_Dt[(self.df_Dt['method'] == 'SEG') & (self.df_Dt['GTV_name'] == 'GTVp')]['Dt'].to_numpy()
        Dt_seg_gtvn = self.df_Dt[(self.df_Dt['method'] == 'SEG') & (self.df_Dt['GTV_name'] == 'GTVn')]['Dt'].to_numpy()
        U, p_Dt_seg_gtv = mannwhitneyu(Dt_seg_gtvp, Dt_seg_gtvn)

        Dt_dnn_gtvp = self.df_Dt[(self.df_Dt['method'] == 'DNN') & (self.df_Dt['GTV_name'] == 'GTVp')]['Dt'].to_numpy()
        Dt_dnn_gtvn = self.df_Dt[(self.df_Dt['method'] == 'DNN') & (self.df_Dt['GTV_name'] == 'GTVn')]['Dt'].to_numpy()
        U, p_Dt_dnn_gtv = mannwhitneyu(Dt_dnn_gtvp, Dt_dnn_gtvn)


        Fp_lsq_gtvp = self.df_Fp[(self.df_Fp['method'] == 'LSQ') & (self.df_Fp['GTV_name'] == 'GTVp')]['Fp'].to_numpy()
        Fp_lsq_gtvn = self.df_Fp[(self.df_Fp['method'] == 'LSQ') & (self.df_Fp['GTV_name'] == 'GTVn')]['Fp'].to_numpy()
        U, p_Fp_conv_gtv = mannwhitneyu(Fp_lsq_gtvp, Fp_lsq_gtvn)

        Fp_seg_gtvp = self.df_Fp[(self.df_Fp['method'] == 'SEG') & (self.df_Fp['GTV_name'] == 'GTVp')]['Fp'].to_numpy()
        Fp_seg_gtvn = self.df_Fp[(self.df_Fp['method'] == 'SEG') & (self.df_Fp['GTV_name'] == 'GTVn')]['Fp'].to_numpy()
        U, p_Fp_seg_gtv = mannwhitneyu(Fp_seg_gtvp, Fp_seg_gtvn)

        Fp_dnn_gtvp = self.df_Fp[(self.df_Fp['method'] == 'DNN') & (self.df_Fp['GTV_name'] == 'GTVp')]['Fp'].to_numpy()
        Fp_dnn_gtvn = self.df_Fp[(self.df_Fp['method'] == 'DNN') & (self.df_Fp['GTV_name'] == 'GTVn')]['Fp'].to_numpy()
        U, p_Fp_dnn_gtv = mannwhitneyu(Fp_dnn_gtvp, Fp_dnn_gtvn)


        Dp_lsq_gtvp = self.df_Dp[(self.df_Dp['method'] == 'LSQ') & (self.df_Dp['GTV_name'] == 'GTVp')]['Dp'].to_numpy()
        Dp_lsq_gtvn = self.df_Dp[(self.df_Dp['method'] == 'LSQ') & (self.df_Dp['GTV_name'] == 'GTVn')]['Dp'].to_numpy()
        U, p_Dp_conv_gtv = mannwhitneyu(Dp_lsq_gtvp, Dp_lsq_gtvn)

        Dp_seg_gtvp = self.df_Dp[(self.df_Dp['method'] == 'SEG') & (self.df_Dp['GTV_name'] == 'GTVp')]['Dp'].to_numpy()
        Dp_seg_gtvn = self.df_Dp[(self.df_Dp['method'] == 'SEG') & (self.df_Dp['GTV_name'] == 'GTVn')]['Dp'].to_numpy()
        U, p_Dp_seg_gtv = mannwhitneyu(Dp_seg_gtvp, Dp_seg_gtvn)

        Dp_dnn_gtvp = self.df_Dp[(self.df_Dp['method'] == 'DNN') & (self.df_Dp['GTV_name'] == 'GTVp')]['Dp'].to_numpy()
        Dp_dnn_gtvn = self.df_Dp[(self.df_Dp['method'] == 'DNN') & (self.df_Dp['GTV_name'] == 'GTVn')]['Dp'].to_numpy()
        U, p_Dp_dnn_gtv = mannwhitneyu(Dp_dnn_gtvp, Dp_dnn_gtvn)

        print(p_Dp_conv_gtv)
        print(p_Dp_seg_gtv)
        print(p_Dp_dnn_gtv)

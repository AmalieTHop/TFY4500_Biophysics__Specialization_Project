
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns
from scipy.integrate import quad
#sns.set()

import os
import copy



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




class Simplot:
    def __init__(self, dst_dir, snrs, reps, num_sims_vis, sim_b_vals):
        self.dst_dir = dst_dir
        self.snrs = snrs
        self.reps = reps
        self.num_sims_vis = num_sims_vis
        self.sim_b_vals = sim_b_vals
        self.b_max = 800
        self.plot_b_vals = np.linspace(0, self.b_max, self.b_max + 1)


        zero_arr = np.zeros((len(self.snrs), self.num_sims_vis))
        self.Dt_gt, self.Fp_gt, self.Dp_gt = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        self.Dt_conv, self.Fp_conv, self.Dp_conv, self.S0_conv = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        self.Dt_seg, self.Fp_seg, self.Dp_seg, self.S0_seg = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        self.Dt_net, self.Fp_net, self.Dp_net, self.S0_net = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        
        self.ivim_signal_noisy = np.zeros((len(self.snrs), self.num_sims_vis, len(self.sim_b_vals)))
        #self.Sb_gt, self.Sb_conv, self.Sb_seg, self.Sb_net = np.zeros((len(self.snrs), self.num_sims_vis, len(self.plot_b_vals))), np.zeros((len(self.snrs), self.num_sims_vis, len(self.plot_b_vals))), np.zeros((len(self.snrs), self.num_sims_vis, len(self.plot_b_vals))), np.zeros((len(self.snrs), self.num_sims_vis, len(self.plot_b_vals)))
        self.exp_estim_err_rmse_conv, self.exp_estim_err_rmse_seg, self.exp_estim_err_rmse_net = np.zeros((len(self.snrs), self.num_sims_vis)), np.zeros((len(self.snrs), self.num_sims_vis)), np.zeros((len(self.snrs), self.num_sims_vis))

        self.color_dim_Dt, self.color_dim_Fp, self.color_dim_Dp = np.zeros((len(self.snrs), 3, self.num_sims_vis)), np.zeros((len(self.snrs), 3, self.num_sims_vis)), np.zeros((len(self.snrs), 3, self.num_sims_vis))
        self.color_dim_exp_estim_err_rmse = np.zeros((len(self.snrs), 3, self.num_sims_vis))


        for i, snr in enumerate(self.snrs):
            src_dir = os.path.join(self.dst_dir, f'SNR{snr}')
            self.Dt_gt[i] = np.load(os.path.join(src_dir, f'Dt_gt_SNR{snr}.npy')).flatten()[:self.num_sims_vis]
            self.Fp_gt[i] = np.load(os.path.join(src_dir, f'Fp_gt_SNR{snr}.npy')).flatten()[:self.num_sims_vis]
            self.Dp_gt[i] = np.load(os.path.join(src_dir, f'Dp_gt_SNR{snr}.npy')).flatten()[:self.num_sims_vis]
            self.Dt_conv[i], self.Fp_conv[i], self.Dp_conv[i], self.S0_conv[i] = np.load(os.path.join(src_dir, f'fitted_params_conv_SNR{snr}.npy'))[:, :self.num_sims_vis]
            self.Dt_seg[i], self.Fp_seg[i], self.Dp_seg[i], self.S0_seg[i] = np.load(os.path.join(src_dir, f'fitted_params_seg_SNR{snr}.npy'))[:, :self.num_sims_vis]
            if reps > 1:
                self.Dt_net[i], self.Fp_net[i], self.Dp_net[i], self.S0_net[i] = np.load(os.path.join(src_dir, f'pred_params_NN_SNR{snr}_reps{self.reps}.npy'))[1, : , :self.num_sims_vis]
            else:
                self.Dt_net[i], self.Fp_net[i], self.Dp_net[i], self.S0_net[i] = np.load(os.path.join(src_dir, f'pred_params_NN_SNR{snr}_reps{self.reps}.npy'))[: , :self.num_sims_vis]
                

            self.ivim_signal_noisy[i] = np.load(os.path.join(src_dir, f'IVIM_signal_noisy_SNR{snr}_reps{self.reps}.npy'))[:self.num_sims_vis, :]    
            
            """
            self.Sb_gt = self.ivim(np.tile(np.expand_dims(self.plot_b_vals, axis=0), (self.num_sims_vis, 1)),
                           np.tile(np.expand_dims(self.Dt_gt[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Fp_gt[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Dp_gt[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(np.ones(self.num_sims_vis), axis=1), (1, len(self.plot_b_vals)))).astype('f')
            self.Sb_conv = self.ivim(np.tile(np.expand_dims(self.plot_b_vals, axis=0), (self.num_sims_vis, 1)),
                           np.tile(np.expand_dims(self.Dt_conv[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Fp_conv[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Dp_conv[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.S0_conv[i], axis=1), (1, len(self.plot_b_vals)))).astype('f')
            self.Sb_seg = self.ivim(np.tile(np.expand_dims(self.plot_b_vals, axis=0), (self.num_sims_vis, 1)),
                           np.tile(np.expand_dims(self.Dt_seg[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Fp_seg[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Dp_seg[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.S0_seg[i], axis=1), (1, len(self.plot_b_vals)))).astype('f')
            self.Sb_net = self.ivim(np.tile(np.expand_dims(self.plot_b_vals, axis=0), (self.num_sims_vis, 1)),
                           np.tile(np.expand_dims(self.Dt_net[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Fp_net[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.Dp_net[i], axis=1), (1, len(self.plot_b_vals))),
                           np.tile(np.expand_dims(self.S0_net[i], axis=1), (1, len(self.plot_b_vals)))).astype('f')

            self.exp_estim_err_rmse_conv = np.sqrt(np.mean(np.square(self.Sb_gt - self.Sb_conv), axis=1))
            self.exp_estim_err_rmse_seg = np.sqrt(np.mean(np.square(self.Sb_gt - self.Sb_seg), axis=1))
            self.exp_estim_err_rmse_net = np.sqrt(np.mean(np.square(self.Sb_gt - self.Sb_net), axis=1))
            """

            for j in range(self.num_sims_vis):
                self.exp_estim_err_rmse_conv[i, j] = np.sqrt(quad(self.ivim_sqared_loss, 0, self.b_max, args=(self.Dt_conv[i, j], self.Fp_conv[i, j], self.Dp_conv[i, j], self.S0_conv[i, j],
                                                                                    self.Dt_gt[i, j], self.Fp_gt[i, j], self.Dp_gt[i, j], 1))[0]/self.b_max)
                self.exp_estim_err_rmse_seg[i, j] = np.sqrt(quad(self.ivim_sqared_loss, 0, self.b_max, args=(self.Dt_seg[i, j], self.Fp_seg[i, j], self.Dp_seg[i, j], self.S0_seg[i, j],
                                                                                    self.Dt_gt[i, j], self.Fp_gt[i, j], self.Dp_gt[i, j], 1))[0]/self.b_max)
                self.exp_estim_err_rmse_net[i, j] = np.sqrt(quad(self.ivim_sqared_loss, 0, self.b_max, args=(self.Dt_net[i, j], self.Fp_net[i, j], self.Dp_net[i, j], self.S0_net[i, j],
                                                                                    self.Dt_gt[i, j], self.Fp_gt[i, j], self.Dp_gt[i, j], 1))[0]/self.b_max)

            self.color_dim_Dt[i] = [(self.Dt_gt[i]-self.Dt_conv[i]), (self.Dt_gt[i] - self.Dt_seg[i]), (self.Dt_gt[i] - self.Dt_net[i])]       #[self.relative_error(self.Dt_gt[i], self.Dt_conv[i]), self.relative_error(self.Dt_gt[i], self.Dt_seg[i]), self.relative_error(self.Dt_gt[i], self.Dt_net[i])]
            self.color_dim_Fp[i] = [(self.Fp_gt[i]-self.Fp_conv[i])*100, (self.Fp_gt[i] - self.Fp_seg[i])*100, (self.Fp_gt[i] - self.Fp_net[i])*100]   #[self.relative_error(self.Fp_gt[i], self.Fp_conv[i]), self.relative_error(self.Fp_gt[i], self.Fp_seg[i]), self.relative_error(self.Fp_gt[i], self.Fp_net[i])]
            self.color_dim_Dp[i] = [(self.Dp_gt[i]-self.Dp_conv[i]), (self.Dp_gt[i] - self.Dp_seg[i]), (self.Dp_gt[i] - self.Dp_net[i])]       #[self.relative_error(self.Dp_gt[i], self.Dp_conv[i]), self.relative_error(self.Dp_gt[i], self.Dp_seg[i]), self.relative_error(self.Dp_gt[i], self.Dp_net[i])]
            
            
            print(self.Fp_seg[(self.Dp_seg > 0.1) & (self.Dp_seg < 0.29)])

            """
            self.color_dim_Dt[i] = [abs(self.Dt_gt[i]-self.Dt_conv[i]), abs(self.Dt_gt[i] - self.Dt_seg[i]), abs(self.Dt_gt[i] - self.Dt_net[i])]       #[self.relative_error(self.Dt_gt[i], self.Dt_conv[i]), self.relative_error(self.Dt_gt[i], self.Dt_seg[i]), self.relative_error(self.Dt_gt[i], self.Dt_net[i])]
            self.color_dim_Fp[i] = [abs(self.Fp_gt[i]-self.Fp_conv[i])*100, abs(self.Fp_gt[i] - self.Fp_seg[i])*100, abs(self.Fp_gt[i] - self.Fp_net[i])*100]   #[self.relative_error(self.Fp_gt[i], self.Fp_conv[i]), self.relative_error(self.Fp_gt[i], self.Fp_seg[i]), self.relative_error(self.Fp_gt[i], self.Fp_net[i])]
            self.color_dim_Dp[i] = [abs(self.Dp_gt[i]-self.Dp_conv[i]), abs(self.Dp_gt[i] - self.Dp_seg[i]), abs(self.Dp_gt[i] - self.Dp_net[i])]       #[self.relative_error(self.Dp_gt[i], self.Dp_conv[i]), self.relative_error(self.Dp_gt[i], self.Dp_seg[i]), self.relative_error(self.Dp_gt[i], self.Dp_net[i])]
            """
            self.color_dim_exp_estim_err_rmse[i] = [self.exp_estim_err_rmse_conv[i], self.exp_estim_err_rmse_seg[i], self.exp_estim_err_rmse_net[i]]
    
    """
    def relative_error(self, y_gt, y_method):
        absolute_error = abs(y_gt - y_method)
        relative_error = absolute_error/y_gt * 100
        return relative_error
    """

    def ivim_sqared_loss(self, b_val, Dt, Fp, Dp, S0, Dt_gt, Fp_gt, Dp_gt, S0_gt = 1):
        return (self.ivim(b_val, Dt_gt, Fp_gt, Dp_gt, S0_gt) - self.ivim(b_val, Dt, Fp, Dp, S0))**2

    def ivim(self, b_val, Dt, Fp, Dp, S0):
        return (S0 * (Fp * np.exp(-b_val * Dp) + (1 - Fp) * np.exp(-b_val * Dt)))


    def plot_signal_curves(self, sim_indices, description = ""):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)

        for i, sim_idx in enumerate(sim_indices):    
            title = f'IVIM signal curves at SNR {self.snrs[i]}' + description

            Sb_gt = self.ivim(self.plot_b_vals, self.Dt_gt[i, sim_idx], self.Fp_gt[i, sim_idx], self.Dp_gt[i, sim_idx], 1)
            Sb_conv = self.ivim(self.plot_b_vals, self.Dt_conv[i, sim_idx], self.Fp_conv[i, sim_idx], self.Dp_conv[i, sim_idx], self.S0_conv[i, sim_idx])
            Sb_seg = self.ivim(self.plot_b_vals, self.Dt_seg[i, sim_idx], self.Fp_seg[i, sim_idx], self.Dp_seg[i, sim_idx], self.S0_seg[i, sim_idx])
            Sb_net = self.ivim(self.plot_b_vals, self.Dt_net[i, sim_idx], self.Fp_net[i, sim_idx], self.Dp_net[i, sim_idx], self.S0_net[i, sim_idx])
                
            plt.figure(figsize = [10,5])
            plt.plot(self.plot_b_vals, Sb_gt, label = "Ground truth", color='C5')
            plt.plot(self.plot_b_vals, Sb_conv, label = "LSQ", color='C0')
            plt.plot(self.plot_b_vals, Sb_seg, label = "SEG", color='C1')
            plt.plot(self.plot_b_vals, Sb_net, label = "DNN", color='C2')
            plt.scatter(self.sim_b_vals, self.ivim_signal_noisy[i, sim_idx], label = "Noisy measured signal", color='C8')
            #plt.title(title, fontsize = 20)
            plt.xlabel(r"$b$-value")
            plt.ylabel(r"$S_{norm}(b)$")
            plt.yticks()
            plt.xticks()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dst_dir, title))



    def plot_estimated_param_map(self, description, color_dim = None, size_dim_indecies = [0]):
        cbar_description = ''
        savename_root = ''
        cbformat = None
        if description == 'Dt':
            cbar_description = r'Absolute error of $D_t$ [mm$^2$/s]'
            savename_root ='abs_err_Dt_'
            cbformat = OOMFormatter(-3, "%1.1f")
        elif description == 'Fp':
            cbar_description = r'Absolute error of $f_p$ [%]'
            savename_root ='abs_err_Fp_'
        elif description == 'Dp':
            cbar_description = r'Absolute error of $D_p$ [mm$^2$/s]'
            savename_root ='abs_err_Dp_'
            cbformat = OOMFormatter(-3, "%1.1f")
        elif description == 'RMSEE':
            cbar_description = r'RMSEE'
            savename_root ='RMSEE_'
        else:
            print('No valid imput for description')
        
        #plt.rcParams.update(plt.rcParamsDefault)
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'image.cmap': 'turbo',
                  'savefig.format': 'pfd'
                  }
        plt.rcParams.update(params)

        alpha = 0.8
        s=15
        color_sims_gt ='darkgray'
        color_sim_gt ='dimgray'


        for snr_idx, snr in enumerate(self.snrs):
            fig, [(ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22), (ax30, ax31, ax32)] = plt.subplots(nrows = 4, ncols = 3, figsize=(20, 20))

            scatter00 = ax00.scatter(x=self.Dt_gt[snr_idx], y=self.Fp_gt[snr_idx]*100, alpha=alpha, c=color_sims_gt, s=s)
            scatter01 = ax01.scatter(x=self.Fp_gt[snr_idx]*100, y=self.Dp_gt[snr_idx], alpha=alpha, c=color_sims_gt, s=s)
            scatter02 = ax02.scatter(x=self.Dp_gt[snr_idx], y=self.Dt_gt[snr_idx], alpha=alpha, c=color_sims_gt, s=s)

            scatter10 = ax10.scatter(x=self.Dt_conv[snr_idx], y=self.Fp_conv[snr_idx]*100, c=color_dim[snr_idx, 0], alpha=alpha, s=s)
            scatter11 = ax11.scatter(x=self.Fp_conv[snr_idx]*100, y=self.Dp_conv[snr_idx], c=color_dim[snr_idx, 0], alpha=alpha, s=s)
            scatter12 = ax12.scatter(x=self.Dp_conv[snr_idx], y=self.Dt_conv[snr_idx], c=color_dim[snr_idx, 0], alpha=alpha, s=s)

            scatter20 = ax20.scatter(x=self.Dt_seg[snr_idx], y=self.Fp_seg[snr_idx]*100, c=color_dim[snr_idx, 1],  alpha=alpha, s=s)
            scatter21 = ax21.scatter(x=self.Fp_seg[snr_idx]*100, y=self.Dp_seg[snr_idx], c=color_dim[snr_idx, 1], alpha=alpha, s=s)
            scatter22 = ax22.scatter(x=self.Dp_seg[snr_idx], y=self.Dt_seg[snr_idx], c=color_dim[snr_idx, 1], alpha=alpha, s=s)
            
            scatter30 = ax30.scatter(x=self.Dt_net[snr_idx], y=self.Fp_net[snr_idx]*100, c=color_dim[snr_idx, 2], alpha=alpha, s=s)
            scatter31 = ax31.scatter(x=self.Fp_net[snr_idx]*100, y=self.Dp_net[snr_idx], c=color_dim[snr_idx, 2], alpha=alpha, s=s)
            scatter32 = ax32.scatter(x=self.Dp_net[snr_idx], y=self.Dt_net[snr_idx], c=color_dim[snr_idx, 2], alpha=alpha, s=s)


            if any(size_dim_indecies):
                size = 500
                v_max_conv = np.max(color_dim[snr_idx, 0])
                v_max_seg = np.max(color_dim[snr_idx, 1])
                v_max_net = np.max(color_dim[snr_idx, 2])

                ax00.scatter(x=[self.Dt_gt[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Fp_gt[snr_idx, size_dim_indecies[snr_idx]]*100], s = [size], alpha=alpha, edgecolors='black', c=color_sim_gt)
                ax01.scatter(x=[self.Fp_gt[snr_idx, size_dim_indecies[snr_idx]]*100], y=[self.Dp_gt[snr_idx, size_dim_indecies[snr_idx]]], s = [size], alpha=alpha, edgecolors='black', c=color_sim_gt)
                ax02.scatter(x=[self.Dp_gt[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Dt_gt[snr_idx, size_dim_indecies[snr_idx]]], s = [size], alpha=alpha, edgecolors='black', c=color_sim_gt)

                ax10.scatter(x=[self.Dt_conv[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Fp_conv[snr_idx, size_dim_indecies[snr_idx]]*100], c=[color_dim[snr_idx, 0, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_conv, s = [size], alpha=alpha, edgecolors='black')
                ax11.scatter(x=[self.Fp_conv[snr_idx, size_dim_indecies[snr_idx]]*100], y=[self.Dp_conv[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 0, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_conv, s = [size], alpha=alpha, edgecolors='black')
                ax12.scatter(x=[self.Dp_conv[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Dt_conv[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 0, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_conv, s = [size], alpha=alpha, edgecolors='black')

                ax20.scatter(x=[self.Dt_seg[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Fp_seg[snr_idx, size_dim_indecies[snr_idx]]*100], c=[color_dim[snr_idx, 1, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_seg, s = [size], alpha=alpha, edgecolors='black')
                ax21.scatter(x=[self.Fp_seg[snr_idx, size_dim_indecies[snr_idx]]*100], y=[self.Dp_seg[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 1, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_seg, s = [size], alpha=alpha, edgecolors='black')
                ax22.scatter(x=[self.Dp_seg[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Dt_seg[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 1, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_seg, s = [size], alpha=alpha, edgecolors='black')

                ax30.scatter(x=[self.Dt_net[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Fp_net[snr_idx, size_dim_indecies[snr_idx]]*100], c=[color_dim[snr_idx, 2, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_net, s = [size], alpha=alpha, edgecolors='black')
                ax31.scatter(x=[self.Fp_net[snr_idx, size_dim_indecies[snr_idx]]*100], y=[self.Dp_net[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 2, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_net, s = [size], alpha=alpha, edgecolors='black')
                ax32.scatter(x=[self.Dp_net[snr_idx, size_dim_indecies[snr_idx]]], y=[self.Dt_net[snr_idx, size_dim_indecies[snr_idx]]], c=[color_dim[snr_idx, 2, size_dim_indecies[snr_idx]]], vmin=0, vmax=v_max_net, s = [size], alpha=alpha, edgecolors='black')

            cbar00 = fig.colorbar(scatter10, ax = ax00, format=cbformat) # dummy
            cbar00.set_label(label='')               # dummy
            cbar01 = fig.colorbar(scatter10, ax = ax01, format=cbformat) # dummy
            cbar01.set_label(label='')               # dummy
            cbar02 = fig.colorbar(scatter10, ax = ax02, format=cbformat) # dummy
            cbar02.set_label(label='')               # dummy
            cbar10 = fig.colorbar(scatter10, ax = ax10, format=cbformat)
            cbar10.set_label(label=fr'{cbar_description}')
            cbar11 = fig.colorbar(scatter11, ax = ax11, format=cbformat)
            cbar11.set_label(label=fr'{cbar_description}')
            cbar12 = fig.colorbar(scatter12, ax = ax12, format=cbformat)
            cbar12.set_label(label=fr'{cbar_description}')
            cbar20 = fig.colorbar(scatter20, ax = ax20, format=cbformat)
            cbar20.set_label(label=fr'{cbar_description}')
            cbar21 = fig.colorbar(scatter21, ax = ax21, format=cbformat)
            cbar21.set_label(label=fr'{cbar_description}')
            cbar22 = fig.colorbar(scatter22, ax = ax22, format=cbformat)
            cbar22.set_label(label=fr'{cbar_description}')
            cbar30 = fig.colorbar(scatter30, ax = ax30, format=cbformat)
            cbar30.set_label(label=fr'{cbar_description}')
            cbar31 = fig.colorbar(scatter31, ax = ax31, format=cbformat)
            cbar31.set_label(label=fr'{cbar_description}')
            cbar33 = fig.colorbar(scatter32, ax = ax32, format=cbformat)
            cbar33.set_label(label=fr'{cbar_description}')

            
            ax00.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='Ground truth')
            ax10.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='LSQ')
            ax20.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='SEG')
            ax30.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='DNN')
            ax00.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax10.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax20.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax30.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            
            ax01.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='Ground truth')
            ax11.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='LSQ')
            ax21.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='SEG')
            ax31.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='DNN')
            ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax11.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax31.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

            ax02.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='Ground truth')
            ax12.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='LSQ')
            ax22.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='SEG')
            ax32.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='DNN')
            ax02.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax12.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax22.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax32.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax12.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax32.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))


            fig.tight_layout(pad= 1.5)
            plt.savefig(os.path.join(self.dst_dir, savename_root + f'SNR{snr}' + '.svg'))
            plt.savefig(os.path.join(self.dst_dir, savename_root + f'SNR{snr}' + '.pdf'))
        
        #plt.rcParams.update(plt.rcParamsDefault)



    def plot_true_param_map(self, description, color_dim = None, size_dim = None):
        cbar_description = ''
        savename_root = ''
        cbformat = None
        if description == 'Dt':
            cbar_description = r'Absolute error of $D_t$ [mm$^2$/s]'
            savename_root ='abs_err_Dt_'
            cbformat = OOMFormatter(-3, "%1.1f")
        elif description == 'Fp':
            cbar_description = r'Absolute error of $f_p$ [%]'
            savename_root ='abs_err_Fp_'
        elif description == 'Dp':
            cbar_description = r'Absolute error of $D_p$ [mm$^2$/s]'
            savename_root ='abs_err_Dp_'
            cbformat = OOMFormatter(-3, "%1.1f")
        elif description == 'RMSEE':
            cbar_description = r'RMSEE'
            savename_root ='RMSEE_'
        else:
            print('No valid imput for description')


        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        
        plt.rcParams.update(params)

        alpha = 0.8
        s=15


        for snr_idx, snr in enumerate(self.snrs):
            fig, [(ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22)] = plt.subplots(nrows = 3, ncols = 3, figsize=(20, 15))

            scatter00 = ax00.scatter(x=self.Dt_gt[snr_idx], y=self.Fp_gt[snr_idx]*100, c=color_dim[snr_idx, 0],alpha=alpha, s=s, vmin=0)
            scatter10 = ax10.scatter(x=self.Dt_gt[snr_idx], y=self.Fp_gt[snr_idx]*100, c=color_dim[snr_idx, 1], alpha=alpha, s=s, vmin=0)
            scatter20 = ax20.scatter(x=self.Dt_gt[snr_idx], y=self.Fp_gt[snr_idx]*100, c=color_dim[snr_idx, 2], alpha=alpha, s=s, vmin=0)

            scatter01 = ax01.scatter(x=self.Fp_gt[snr_idx]*100, y=self.Dp_gt[snr_idx], c=color_dim[snr_idx, 0], alpha=alpha, s=s, vmin=0)
            scatter11 = ax11.scatter(x=self.Fp_gt[snr_idx]*100, y=self.Dp_gt[snr_idx], c=color_dim[snr_idx, 1], alpha=alpha, s=s, vmin=0)
            scatter21 = ax21.scatter(x=self.Fp_gt[snr_idx]*100, y=self.Dp_gt[snr_idx], c=color_dim[snr_idx, 2], alpha=alpha, s=s, vmin=0)

            scatter02 = ax02.scatter(x=self.Dp_gt[snr_idx], y=self.Dt_gt[snr_idx], c=color_dim[snr_idx, 0], alpha=alpha, s=s, vmin=0)
            scatter12 = ax12.scatter(x=self.Dp_gt[snr_idx], y=self.Dt_gt[snr_idx], c=color_dim[snr_idx, 1], alpha=alpha, s=s, vmin=0)
            scatter22 = ax22.scatter(x=self.Dp_gt[snr_idx], y=self.Dt_gt[snr_idx], c=color_dim[snr_idx, 2], alpha=alpha, s=s, vmin=0)

            cbar00 = fig.colorbar(scatter00, ax = ax00, format=cbformat)
            cbar00.set_label(label=fr'{cbar_description}')
            cbar10 = fig.colorbar(scatter10, ax = ax10, format=cbformat)
            cbar10.set_label(label=fr'{cbar_description}')
            cbar20 = fig.colorbar(scatter20, ax = ax20, format=cbformat)
            cbar20.set_label(label=fr'{cbar_description}')

            cbar01 = fig.colorbar(scatter01, ax = ax01, format=cbformat)
            cbar01.set_label(label=fr'{cbar_description}')
            cbar11 = fig.colorbar(scatter11, ax = ax11, format=cbformat)
            cbar11.set_label(label=fr'{cbar_description}')
            cbar21 = fig.colorbar(scatter21, ax = ax21, format=cbformat)
            cbar21.set_label(label=fr'{cbar_description}')

            cbar02 = fig.colorbar(scatter02, ax = ax02, format=cbformat)
            cbar02.set_label(label=fr'{cbar_description}')
            cbar12 = fig.colorbar(scatter12, ax = ax12, format=cbformat)
            cbar12.set_label(label=fr'{cbar_description}')
            cbar22 = fig.colorbar(scatter22, ax = ax22, format=cbformat)
            cbar22.set_label(label=fr'{cbar_description}')


            ax00.set(xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='LSQ')
            ax10.set(xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='SEG')
            ax20.set(xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='DNN')
            ax00.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax10.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax20.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            
            ax01.set(xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='LSQ')
            ax11.set(xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='SEG')
            ax21.set(xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='DNN')
            ax01.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax11.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax21.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

            ax02.set(xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='LSQ')
            ax12.set(xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='SEG')
            ax22.set(xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='DNN')
            ax02.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax12.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax22.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax02.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax12.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
            ax22.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))

            fig.tight_layout()
            plt.savefig(os.path.join(self.dst_dir, savename_root + f'SNR{snr}' + ' - true.svg'))
            plt.savefig(os.path.join(self.dst_dir, savename_root + f'SNR{snr}' + ' - true.pdf'))



    def plot_sim_stat(self):
        
        df_nrmse = pd.DataFrame(columns=['SNR', 'param', 'Method', 'param_val'])
        df_rho = pd.DataFrame(columns=['SNR', 'p', 'Method', 'p_val'])
        df_cv = pd.DataFrame(columns=['SNR', 'param', 'Method', 'cv_val'])

        snrs = [str(x) for x in self.snrs] #[8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
        for snr in snrs:
            src_dir = os.path.join(self.dst_dir, f'SNR{snr}')

            mat_conv = np.load(os.path.join(src_dir, f'mat_conv_SNR{snr}.npy'))
            mat_seg = np.load(os.path.join(src_dir, f'mat_seg_SNR{snr}.npy'))
            mat_net = np.load(os.path.join(src_dir, f'mat_NN_SNR{snr}_reps{self.reps}_raw.npy'))
            stability_net = np.load(os.path.join(src_dir, f'stability_SNR{snr}_reps{self.reps}.npy'))

            nrmse_conv_Dt = pd.DataFrame({'SNR': [snr], 'param': 'Dt', 'Method': 'LSQ', 'param_val': [mat_conv[1,1]]})
            nrmse_conv_Fp = pd.DataFrame({'SNR': [snr], 'param': 'Fp', 'Method': 'LSQ', 'param_val': [mat_conv[0,1]]})
            nrmse_conv_Dp = pd.DataFrame({'SNR': [snr], 'param': 'Dp', 'Method': 'LSQ', 'param_val': [mat_conv[2,1]]})
            sp_conv_Dt = pd.DataFrame({'SNR': [snr], 'p': 'DtDp', 'Method': 'LSQ', 'p_val': [mat_conv[0,2]]})
            sp_conv_Fp = pd.DataFrame({'SNR': [snr], 'p': 'DtFp', 'Method': 'LSQ', 'p_val': [mat_conv[1,2]]})
            sp_conv_Dp = pd.DataFrame({'SNR': [snr], 'p': 'DpFp', 'Method': 'LSQ', 'p_val': [mat_conv[2,2]]})

            nrmse_seg_Dt = pd.DataFrame({'SNR': [snr], 'param': 'Dt', 'Method': 'SEG', 'param_val': [mat_seg[1,1]]})
            nrmse_seg_Fp = pd.DataFrame({'SNR': [snr], 'param': 'Fp', 'Method': 'SEG', 'param_val': [mat_seg[0,1]]})
            nrmse_seg_Dp = pd.DataFrame({'SNR': [snr], 'param': 'Dp', 'Method': 'SEG', 'param_val': [mat_seg[2,1]]})
            sp_seg_Dt = pd.DataFrame({'SNR': [snr], 'p': 'DtDp', 'Method': 'SEG', 'p_val': [mat_seg[0,2]]})
            sp_seg_Fp = pd.DataFrame({'SNR': [snr], 'p': 'DtFp', 'Method': 'SEG', 'p_val': [mat_seg[1,2]]})
            sp_seg_Dp = pd.DataFrame({'SNR': [snr], 'p': 'DpFp', 'Method': 'SEG', 'p_val': [mat_seg[2,2]]})

            df_nrmse = pd.concat([df_nrmse, nrmse_conv_Dt, nrmse_conv_Fp, nrmse_conv_Dp, 
                                            nrmse_seg_Dt, nrmse_seg_Fp, nrmse_seg_Dp])
            df_rho = pd.concat([df_rho, sp_conv_Dt, sp_conv_Fp, sp_conv_Dp, 
                                    sp_seg_Dt, sp_seg_Fp, sp_seg_Dp]) 

            for i in range(self.reps):
                nrmse_net_Dt = pd.DataFrame({'SNR': [snr], 'param': 'Dt', 'Method': 'DNN', 'param_val': [mat_net[i,1,1]]})
                nrmse_net_Fp = pd.DataFrame({'SNR': [snr], 'param': 'Fp', 'Method': 'DNN', 'param_val': [mat_net[i,0,1]]})
                nrmse_net_Dp = pd.DataFrame({'SNR': [snr], 'param': 'Dp', 'Method': 'DNN', 'param_val': [mat_net[i,2,1]]})
                sp_net_Dt = pd.DataFrame({'SNR': [snr], 'p': 'DtDp', 'Method': 'DNN', 'p_val': [mat_net[i,0,2]]})
                sp_net_Fp = pd.DataFrame({'SNR': [snr], 'p': 'DtFp', 'Method': 'DNN', 'p_val': [mat_net[i,1,2]]})
                sp_net_Dp = pd.DataFrame({'SNR': [snr], 'p': 'DpFp', 'Method': 'DNN', 'p_val': [mat_net[i,2,2]]})
            
                df_nrmse = pd.concat([df_nrmse, nrmse_net_Dt, nrmse_net_Fp, nrmse_net_Dp])
                df_rho = pd.concat([df_rho, sp_net_Dt, sp_net_Fp, sp_net_Dp])


            cv_net_Dt = pd.DataFrame({'SNR': [snr], 'param': 'Dt', 'Method': 'DNN', 'cv_val': [stability_net[0]]})
            cv_net_Fp = pd.DataFrame({'SNR': [snr], 'param': 'Fp', 'Method': 'DNN', 'cv_val': [stability_net[1]]})
            cv_net_Dp = pd.DataFrame({'SNR': [snr], 'param': 'Dp', 'Method': 'DNN', 'cv_val': [stability_net[2]]})

            df_cv = pd.concat([df_cv, cv_net_Dt, cv_net_Fp, cv_net_Dp])

        df_nrmse.to_csv(self.dst_dir + '/df_nrmse', index=False)
        df_rho.to_csv(self.dst_dir + '/df_rho', index=False)
        df_cv.to_csv(self.dst_dir + '/df_cv', index=False)


        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 20,
                  'legend.loc':'best', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        
        plt.rcParams.update(params)
        estimator = np.median
        capsize=0.1
        errorbar=('pi', 50)

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        ax00 = sns.pointplot(data=df_nrmse[df_nrmse.param == 'Dt'], x='SNR', y='param_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[0,0])
        ax10 = sns.pointplot(data=df_nrmse[df_nrmse.param == 'Fp'], x='SNR', y='param_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[1,0])
        ax20 = sns.pointplot(data=df_nrmse[df_nrmse.param == 'Dp'], x='SNR', y='param_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[2,0])

        ax01 = sns.pointplot(data=df_rho[df_rho.p == 'DtDp'], x='SNR', y='p_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[0,1])
        ax11 = sns.pointplot(data=df_rho[df_rho.p == 'DtFp'], x='SNR', y='p_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[1,1])
        ax21 = sns.pointplot(data=df_rho[df_rho.p == 'DpFp'], x='SNR', y='p_val', hue='Method', estimator=estimator, errorbar=errorbar, capsize=capsize, markers=['o', 'v', 'D'], ax=axes[2,1])

        ax02 = sns.pointplot(data=df_cv[df_cv.param == 'Dt'], x='SNR', y='cv_val', hue='Method', estimator=estimator, markers='D', ax=axes[0,2], palette=['C2'])
        ax12 = sns.pointplot(data=df_cv[df_cv.param == 'Fp'], x='SNR', y='cv_val', hue='Method', estimator=estimator, markers='D', ax=axes[1,2], palette=['C2'])
        ax22 = sns.pointplot(data=df_cv[df_cv.param == 'Dp'], x='SNR', y='cv_val', hue='Method', estimator=estimator, markers='D', ax=axes[2,2], palette=['C2'])

        """
        # if len(bvals)==5
        ax00.set(xticks=snrs, ylim=(0, 0.75), ylabel = r'NRMSE of $D_t$ [fractional]')
        ax01.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_t$, $D_p$)')
        ax02.set(xticks=snrs, ylim=(0, 0.1), ylabel = r'CV of $D_t$ [fractional]')

        ax10.set(xticks=snrs, ylim=(0, 0.75), ylabel = r'NRMSE of $f_p$ [fractional]')
        ax11.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_t$, $f_p$)')
        ax12.set(xticks=snrs, ylim=(0, 0.1), ylabel = r'CV of $f_p$ [fractional]')

        ax20.set(xticks=snrs, ylim=(0, 3), ylabel = r'NRMSE of $D_p$ [fractional]')
        ax21.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_p$, $f_p$)')
        ax22.set(xticks=snrs, ylim=(0, 1), ylabel = r'CV of $D_p$ [fractional]')
        """
        
        # if len(bvals)==11
        ax00.set(xticks=snrs, ylim=(0, 0.75), ylabel = r'NRMSE of $D_t$ [fractional]')
        ax01.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_t$, $D_p$)')
        ax02.set(xticks=snrs, ylim=(0, 0.1), ylabel = r'CV of $D_t$ [fractional]')

        ax10.set(xticks=snrs, ylim=(0, 0.75), ylabel = r'NRMSE of $f_p$ [fractional]')
        ax11.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_t$, $f_p$)')
        ax12.set(xticks=snrs, ylim=(0, 0.1), ylabel = r'CV of $f_p$ [fractional]')

        ax20.set(xticks=snrs, ylim=(0, 1.75), ylabel = r'NRMSE of $D_p$ [fractional]')
        ax21.set(xticks=snrs, ylim=(0, 1), ylabel = r'$\rho$($D_p$, $f_p$)')
        ax22.set(xticks=snrs, ylim=(0, 0.1), ylabel = r'CV of $D_p$ [fractional]')
        


        ax00.legend(title=None); ax01.legend(title=None); ax02.legend(title=None)
        ax10.legend(title=None); ax11.legend(title=None); ax12.legend(title=None)
        ax20.legend(title=None); ax21.legend(title=None); ax22.legend(title=None)

        fig.tight_layout()
        #plt.savefig(os.path.join(self.dst_dir, f'stats_from_sims_reps{self.reps}.svg'))  
        plt.savefig(os.path.join(self.dst_dir, f'stats_from_sims_reps{self.reps}.pdf'))    


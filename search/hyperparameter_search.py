import numpy as np
import pandas as pd

from behavenet.fitting.utils import get_best_model_and_data
from behavenet.plotting import (
    load_latents,
    load_metrics_csv_as_df
)

from psvae_experiment import PSvaeExperiment
from search_utils import get_label_r2


def alpha_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha_weights, beta, gamma,
                 **kwargs):
    metrics_list = ['loss_data_mse']
    metrics_dfs_frame, metrics_dfs_marker = [], []

    for alpha_ in alpha_weights:
        try:
            a_search = PSvaeExperiment(
                lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha_, beta, gamma, **kwargs)
        except TypeError:
            print('Could not find model for alpha=%i, beta=%i, gamma=%i' % (
                alpha_, beta, gamma))
            continue

        hparams = a_search.hparams
        print('Loading results with alpha=%i, beta=%i, gamma=%i (version %i)' % (
            hparams['ps_vae.alpha'], hparams['ps_vae.beta'], hparams['ps_vae.gamma'],
            a_search.version))
        # get frame mse
        metrics_dfs_frame.append(load_metrics_csv_as_df(
            hparams, lab, expt, metrics_list, version=None, test=True))
        metrics_dfs_frame[-1]['alpha'] = alpha_
        metrics_dfs_frame[-1]['n_latents'] = hparams['n_ae_latents']
        # get marker mse
        model, data_gen = get_best_model_and_data(
            hparams, Model=None, load_data=True, version=a_search.version)
        metrics_df_ = get_label_r2(
            hparams, model, data_gen, a_search.version, a_search.label_names, dtype='val')
        metrics_df_['alpha'] = alpha_
        metrics_df_['n_latents'] = hparams['n_ae_latents']
        metrics_dfs_marker.append(
            metrics_df_[metrics_df_.Model == 'PS-VAE'])

    try:
        metrics_df_frame = pd.concat(metrics_dfs_frame, sort=False)
        metrics_df_marker = pd.concat(metrics_dfs_marker, sort=False)
        return metrics_df_frame, metrics_df_marker
    except ValueError:
        print("No experiments were found for alpha hyperparameter search")


def beta_gamma_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta_weights,
                      gamma_weights, batch_size=None, **kwargs):
    metrics_list = ['loss_data_mse', 'loss_zu_mi', 'loss_zu_tc', 'loss_zu_dwkl', 'loss_AB_orth']
    metrics_dfs_frame_bg, metrics_dfs_marker_bg, metrics_dfs_corr_bg = [], [], []
    overlaps = {}
    print("searching over b = {}, g = {}".format(beta_weights, gamma_weights))
    for beta in beta_weights:
        for gamma in gamma_weights:
            try:
                bg_search = PSvaeExperiment(
                    lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta, gamma, **kwargs)
                hparams = bg_search.hparams
                print('Loading results with alpha=%i, beta=%i, gamma=%i (version %i)' % (
                    hparams['ps_vae.alpha'], hparams['ps_vae.beta'], hparams['ps_vae.gamma'],
                    bg_search.version))
                # get frame mse
                metrics_dfs_frame_bg.append(load_metrics_csv_as_df(
                    hparams, lab, expt, metrics_list, version=None, test=True))
                metrics_dfs_frame_bg[-1]['beta'] = beta
                metrics_dfs_frame_bg[-1]['gamma'] = gamma
                # get marker mse
                model, data_gen = get_best_model_and_data(
                    hparams, Model=None, load_data=True, version=bg_search.version)
                metrics_df_ = get_label_r2(hparams, model, data_gen, bg_search.version, bg_search.label_names,
                                           dtype='val')
                metrics_df_['beta'] = beta
                metrics_df_['gamma'] = gamma
                metrics_dfs_marker_bg.append(metrics_df_[metrics_df_.Model == 'PS-VAE'])
                # get subspace overlap
                A = model.encoding.A.weight.data.cpu().detach().numpy()
                B = model.encoding.B.weight.data.cpu().detach().numpy()
                C = np.concatenate([A, B], axis=0)
                overlap = np.matmul(C, C.T)
                overlaps['beta=%i_gamma=%i' % (beta, gamma)] = overlap
                # get corr
                latents = load_latents(hparams, bg_search.version, dtype='test')
                if batch_size is None:
                    corr = np.corrcoef(latents[:, bg_search.n_labels + np.array([0, 1])].T)
                    metrics_dfs_corr_bg.append(pd.DataFrame({
                        'loss': 'corr',
                        'dtype': 'test',
                        'val': np.abs(corr[0, 1]),
                        'beta': beta,
                        'gamma': gamma}, index=[0]))
                else:
                    n_batches = int(np.ceil(latents.shape[0] / batch_size))
                    for i in range(n_batches):
                        corr = np.corrcoef(
                            latents[i * batch_size:(i + 1) * batch_size, bg_search.n_labels + np.array([0, 1])].T)
                        metrics_dfs_corr_bg.append(pd.DataFrame({
                            'loss': 'corr',
                            'dtype': 'test',
                            'val': np.abs(corr[0, 1]),
                            'beta': beta,
                            'gamma': gamma}, index=[0]))
            except TypeError:
                print('Could not find model for alpha=%i, beta=%i, gamma=%i' % (
                    alpha, beta, gamma))
                continue

    try:
        metrics_df_frame_bg = pd.concat(metrics_dfs_frame_bg, sort=False)
        metrics_df_marker_bg = pd.concat(metrics_dfs_marker_bg, sort=False)
        metrics_df_corr_bg = pd.concat(metrics_dfs_corr_bg, sort=False)
        return metrics_df_frame_bg, metrics_df_marker_bg, metrics_df_corr_bg
    except ValueError:
        print("No experiments were found for beta/gamma hyperparameter search")


def hyperparameter_search(lab, expt, animal, session, label_names, alpha_expt_name, alpha_n_ae_latents, alpha_weights,
                          beta_weights, gamma_weights, beta=1, gamma=1000, bg_expt_name=None, bg_n_ae_latents=None,
                          batch_size=None, **kwargs):
    # alpha search
    try:
        _, metrics_df_marker = alpha_search(lab, expt, animal, session, label_names, alpha_expt_name,
                                            alpha_n_ae_latents,
                                            alpha_weights, beta, gamma, **kwargs)
    except TypeError:
        print("Exiting hyperparameter search")
        return

    alpha_label_mse = metrics_df_marker[['alpha', 'MSE', 'Label']].groupby(['alpha', 'Label']).median()
    alpha_mse = alpha_label_mse.groupby(['alpha']).median().reset_index()
    max_alpha_idx = alpha_mse[alpha_mse.columns[0]].idxmax()
    baseline_MSE = float(alpha_mse.iloc[max_alpha_idx][['MSE']])

    prev_row = None
    optimal_alpha = None
    for index, row in alpha_mse.iloc[::-1].iterrows():
        if float(row[['MSE']]) > 1.1 * baseline_MSE:
            optimal_alpha = int(prev_row[['alpha']])
            break
        prev_row = row
    print("Alpha found: {}".format(optimal_alpha))

    # beta-gamma search
    if bg_n_ae_latents is None:
        bg_n_ae_latents = alpha_n_ae_latents

    if bg_expt_name is None:
        bg_expt_name = alpha_expt_name
    try:
        _, _, metrics_df_corr_bg = beta_gamma_search(lab, expt, animal, session, label_names, bg_expt_name,
                                                     bg_n_ae_latents, optimal_alpha, beta_weights, gamma_weights,
                                                     batch_size=batch_size, **kwargs)
    except TypeError:
        print("Exiting hyperparameter search")
        return

    metrics_df_corr_bg = metrics_df_corr_bg.reset_index()
    min_corr_idx = metrics_df_corr_bg[['val']].idxmin()
    optimal_beta = int(metrics_df_corr_bg.iloc[min_corr_idx][['beta']].to_numpy().flatten()[0])
    optimal_gamma = int(metrics_df_corr_bg.iloc[min_corr_idx][['gamma']].to_numpy().flatten()[0])
    print("Beta found: {}".format(optimal_beta))
    print("Gamma found: {}".format(optimal_gamma))
    return optimal_alpha, optimal_beta, optimal_gamma

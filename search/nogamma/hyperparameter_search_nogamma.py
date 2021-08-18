import numpy as np
import pandas as pd
from nogamma.psvae_experiment_nogamma import PSvaeExperiment
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.plotting import (
    load_latents,
    load_metrics_csv_as_df
)
from nogamma.search_utils_nogamma import get_label_r2


def alpha_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha_weights, beta,
                 **kwargs):
    metrics_list = ['loss_data_mse']
    metrics_dfs_frame, metrics_dfs_marker = [], []

    for alpha_ in alpha_weights:
        try:
            a_search = PSvaeExperiment(
                lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha_, beta, **kwargs)
        except TypeError:
            print('Could not find model for alpha=%i, beta=%i' % (
                alpha_, beta))
            continue

        hparams = a_search.hparams
        print('Loading results with alpha=%i, beta=%i (version %i)' % (
            hparams['ps_vae.alpha'], hparams['ps_vae.beta'], a_search.version))
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


def beta_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta_weights,
                batch_size=None, **kwargs):
    metrics_list = ['loss_data_mse', 'loss_zu_mi', 'loss_zu_tc', 'loss_zu_dwkl']
    metrics_dfs_frame_b, metrics_dfs_marker_b, metrics_dfs_corr_b = [], [], []
    overlaps = {}
    print("searching over b = {}".format(beta_weights))
    for beta in beta_weights:
        try:
            b_search = PSvaeExperiment(
                lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta, **kwargs)
            hparams = b_search.hparams
            print('Loading results with alpha=%i, beta=%i (version %i)' % (
                hparams['ps_vae.alpha'], hparams['ps_vae.beta'], b_search.version))
            # get frame mse
            metrics_dfs_frame_b.append(load_metrics_csv_as_df(
                hparams, lab, expt, metrics_list, version=None, test=True))
            metrics_dfs_frame_b[-1]['beta'] = beta
            # get marker mse
            model, data_gen = get_best_model_and_data(
                hparams, Model=None, load_data=True, version=b_search.version)
            metrics_df_ = get_label_r2(hparams, model, data_gen, b_search.version, b_search.label_names,
                                       dtype='val')
            metrics_df_['beta'] = beta
            metrics_dfs_marker_b.append(metrics_df_[metrics_df_.Model == 'PS-VAE'])
            # get corr
            latents = load_latents(hparams, b_search.version, dtype='test')
            if batch_size is None:
                corr = np.corrcoef(latents[:, b_search.n_labels + np.array([0, 1])].T)
                metrics_dfs_corr_b.append(pd.DataFrame({
                    'loss': 'corr',
                    'dtype': 'test',
                    'val': np.abs(corr[0, 1]),
                    'beta': beta}, index=[0]))

        except TypeError:
            print('Could not find model for alpha=%i, beta=%i' % (
                alpha, beta))
            continue

    try:
        metrics_df_frame_b = pd.concat(metrics_dfs_frame_b, sort=False)
        metrics_df_marker_b = pd.concat(metrics_dfs_marker_b, sort=False)
        metrics_df_corr_b = pd.concat(metrics_dfs_corr_b, sort=False)
        return metrics_df_frame_b, metrics_df_marker_b, metrics_df_corr_b
    except ValueError:
        print("No experiments were found for beta hyperparameter search")


def hyperparameter_search(lab, expt, animal, session, label_names, alpha_expt_name, alpha_n_ae_latents, alpha_weights,
                          beta_weights, beta=1, beta_expt_name=None, beta_n_ae_latents=None, batch_size=None, **kwargs):
    # alpha search
    try:
        _, metrics_df_marker = alpha_search(lab, expt, animal, session, label_names, alpha_expt_name,
                                            alpha_n_ae_latents, alpha_weights, beta, **kwargs)
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

    # beta search
    if beta_n_ae_latents is None:
        beta_n_ae_latents = alpha_n_ae_latents

    if beta_expt_name is None:
        beta_expt_name = alpha_expt_name
    try:
        _, _, metrics_df_corr_b = beta_search(lab, expt, animal, session, label_names, beta_expt_name,
                                              beta_n_ae_latents, optimal_alpha, beta_weights, batch_size=batch_size,
                                              **kwargs)
    except TypeError:
        print("Exiting hyperparameter search")
        return

    metrics_df_corr_b = metrics_df_corr_b.reset_index()
    min_corr_idx = metrics_df_corr_b[['val']].idxmin()
    optimal_beta = int(metrics_df_corr_b.iloc[min_corr_idx][['beta']].to_numpy().flatten()[0])
    print("Beta found: {}".format(optimal_beta))
    return optimal_alpha, optimal_beta

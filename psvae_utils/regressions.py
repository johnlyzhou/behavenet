import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm

from behavenet import get_user_dir
from behavenet import make_dir_if_not_exists
from behavenet.fitting.utils import experiment_exists
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_lab_example
from behavenet.fitting.utils import get_session_dir
from behavenet.fitting.cond_ae_utils import apply_masks, collect_data


def get_predicted_labels(lrs, latents):
    y_pred = []
    for lr in lrs:
        y_pred.append(lr.predict(latents)[:, None])
    return np.hstack(y_pred)


def fit_regression(model, data_generator, label_names, dtype='val', fit_full=False):
    """Fit regression model from latent space to markers."""

    from sklearn.linear_model import RidgeCV as Ridge

    n_labels = len(label_names)

    print('collecting training labels and latents')
    ys_tr, zs_tr, masks_tr, trials_tr, sessions_tr = collect_data(
        data_generator, model, dtype='train', fit_full=fit_full)
    print('done')

    print('collecting %s labels and latents' % dtype)
    ys, zs, masks, trials, sessions = collect_data(
        data_generator, model, dtype=dtype, fit_full=fit_full)
    print('done')

    print('fitting linear regression model with training data')
    ys_mat = np.concatenate(ys_tr, axis=0)
    zs_mat = np.concatenate(zs_tr, axis=0)
    masks_mat = np.concatenate(masks_tr, axis=0)
    lrs = []
    for i in range(n_labels):
        print('label %i/%i' % (i + 1, n_labels))
        lrs.append(Ridge(alphas=(0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), cv=5).fit(
            apply_masks(zs_mat, masks_mat[:, i]), apply_masks(ys_mat[:, i], masks_mat[:, i])))
    print('done')
    # y_baseline = np.mean(ys_mat, axis=0)
    y_baseline = np.array(
        [np.mean(apply_masks(ys_mat[:, i], masks_mat[:, i]), axis=0) for i in range(n_labels)])

    print('computing r2 on %s data' % dtype)
    metrics_df = []
    for i_test in tqdm(range(data_generator.n_tot_batches[dtype])):
        for i in range(n_labels):
            y_true = apply_masks(ys[i_test][:, i], masks[i_test][:, i])
            if len(y_true) > 10:
                y_pred = lrs[i].predict(apply_masks(zs[i_test], masks[i_test][:, i]))
                r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
                mse = np.mean(np.square(y_true - y_pred))
            else:
                r2 = np.nan
                mse = np.nan
            metrics_df.append(pd.DataFrame({
                'Trial': trials[i_test],
                'Session': sessions[i_test][0],
                'Label': label_names[i],
                'R2': r2,
                'MSE': mse,
                'Model': model.hparams['model_class']}, index=[0]))
            mse_base = np.mean(np.square(y_true - y_baseline[i]))
            metrics_df.append(pd.DataFrame({
                'Trial': trials[i_test],
                'Session': sessions[i_test][0],
                'Label': label_names[i],
                'R2': 0,
                'MSE': mse_base,
                'Model': 'baseline'}, index=[0]))
    print('done')

    return pd.concat(metrics_df, sort=True), lrs


def compute_r2(
        type, hparams, model, data_generator, version, label_names, dtype='val', overwrite=False,
        save_results=True):
    """

    Parameters
    ----------
    type
        'supervised' | 'unsupervised' | 'full'
    hparams
    model
    data_generator
    version
    label_names
    dtype
    overwrite
    save_results

    Returns
    -------

    """

    n_labels = len(label_names)
    save_file = os.path.join(hparams['expt_dir'], 'version_%i' % version, 'r2_%s.csv' % type)
    if type == 'unsupervised':
        model_file = os.path.join(os.path.dirname(save_file), 'regressions.pkl')
    elif type == 'full':
        model_file = os.path.join(os.path.dirname(save_file), 'regressions_full.pkl')
    else:
        model_file = None

    if not os.path.exists(save_file) or overwrite:

        if not os.path.exists(save_file):
            print('R^2 metrics do not exist; computing from scratch')
        else:
            print('overwriting metrics at %s' % save_file)

        if type == 'supervised':
            metrics_df = []
            lrs = None
            data_generator.reset_iterators(dtype)
            for i_test in tqdm(range(data_generator.n_tot_batches[dtype])):
                # get next minibatch and put it on the device
                data, sess = data_generator.next_batch(dtype)
                x = data['images'][0]
                y = data['labels'][0].cpu().detach().numpy()
                if 'labels_masks' in data:
                    n = data['labels_masks'][0].cpu().detach().numpy()
                else:
                    n = np.ones_like(y)
                z = model.get_transformed_latents(x, dataset=sess)
                for i in range(n_labels):
                    y_true = apply_masks(y[:, i], n[:, i])
                    y_pred = apply_masks(z[:, i], n[:, i])
                    if len(y_true) > 10:
                        r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
                        mse = np.mean(np.square(y_true - y_pred))
                    else:
                        r2 = np.nan
                        mse = np.nan
                    metrics_df.append(pd.DataFrame({
                        'Trial': data['batch_idx'].item(),
                        'Session': sess,
                        'Label': label_names[i],
                        'R2': r2,
                        'MSE': mse,
                        'Model': model.hparams['model_class']}, index=[0]))
            metrics_df = pd.concat(metrics_df)
        elif type == 'unsupervised':
            metrics_df, lrs = fit_regression(
                model, data_generator, label_names, dtype=dtype)
        elif type == 'full':
            metrics_df, lrs = fit_regression(
                model, data_generator, label_names, dtype=dtype, fit_full=True)
        else:
            raise NotImplementedError
        print('done')
        if save_results:
            print('saving results to %s' % save_file)
            metrics_df.to_csv(save_file, index=False, header=True)
            if type == 'unsupervised' or type == 'full':
                print('saving models to %s' % model_file)
                with open(model_file, 'wb') as f:
                    pickle.dump(lrs, f)
                
    else:
        print('loading results from %s' % save_file)
        metrics_df = pd.read_csv(save_file)
        if model_file is not None:
            print('loading regression models from %s' % model_file)
            with open(model_file, 'rb') as f:
                lrs = pickle.load(f)
        else:
            lrs = None

    return metrics_df, lrs


def plot_reconstruction_traces(
        traces, names, save_file=None, xtick_locs=None, frame_rate=None, format='png',
        scale=0.5, max_traces=8, add_r2=False, add_legend=True):
    """Plot latents and their neural reconstructions.

    Parameters
    ----------
    traces : :obj:`list`
        each entry is of shape (n_frames, n_dims)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    xtick_locs : :obj:`array-like`, optional
        tick locations in units of bins
    frame_rate : :obj:`float`, optional
        frame rate of behavorial video; to properly relabel xticks
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'
    scale : :obj:`int`, optional
        scale magnitude of traces
    max_traces : :obj:`int`, optional
        maximum number of traces to plot, for easier visualization
    add_r2 : :obj:`bool`, optional
        print R2 value on plot

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle

    """

    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import seaborn as sns

    sns.set_style('white')
    sns.set_context('poster')

    assert len(traces) == len(names)

    means = np.nanmean(traces[0], axis=0)
    stds = np.nanstd(traces[0]) / scale  # scale for better visualization
    for m, mean in enumerate(means):
        if np.isnan(mean):
            means[m] = np.nanmean(traces[1][:, m])

    traces_sc = []
    for trace in traces:
        traces_sc.append((trace - means) / stds)

    fig = plt.figure(figsize=(12, 8))  # (12, 6) for ps-vae paper
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:(len(traces) - 1)]
    colors.insert(0, '#000000')
    linewidths = [2] * len(colors)
    linewidths[0] = 4
    for trace_sc, color, linewidth in zip(traces_sc, colors, linewidths):
        plt.plot(trace_sc + np.arange(trace_sc.shape[1]), linewidth=linewidth, color=color)

    # add legend
    if add_legend:
        lines = []
        for color in colors:
            lines.append(mlines.Line2D([], [], color=color, linewidth=3, alpha=0.7))
        plt.legend(
            lines, names, loc='lower right', frameon=True, framealpha=0.7, edgecolor=[1, 1, 1])

    # if add_r2:
    #     from sklearn.metrics import r2_score
    #     r2 = r2_score(traces_ae, traces_neural, multioutput='variance_weighted')
    #     plt.text(
    #         0.05, 0.06, '$R^2$=%1.3f' % r2, horizontalalignment='left', verticalalignment='bottom',
    #         transform=plt.gca().transAxes,
    #         bbox=dict(facecolor='white', alpha=0.7, edgecolor=[1, 1, 1]))

    if xtick_locs is not None and frame_rate is not None:
        if xtick_locs[0] / frame_rate < 1:
            plt.xticks(xtick_locs, (np.asarray(xtick_locs) / frame_rate))
        else:
            plt.xticks(xtick_locs, (np.asarray(xtick_locs) / frame_rate).astype('int'))
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Time (bins)')
    plt.ylabel('Latent state')
    plt.yticks([])

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    plt.show()
    return fig


def plot_label_latent_regressions(
        lab, expt, animal, session, alpha, beta, gamma, n_ae_latents, rng_seed_model,
        label_names, sssvae_experiment_name, vae_experiment_name, delta=None,
        models=['sss-vae-s', 'vae'],
        dtype='test', measure='r2', save_results=True, overwrite=False,
        save_file=None, format='pdf', **kwargs):

    # ------------------------------------------
    # perform regressions
    # ------------------------------------------
    metrics_df = {}
    for model in models:
        print()
        print(model)

        # get model
        if model == 'vae':
            hparams = _get_psvae_hparams(
                model_class='ps-vae', alpha=alpha, beta=beta, gamma=gamma,
                n_ae_latents=n_ae_latents,
                experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)
            hparams_vae = copy.deepcopy(hparams)
            hparams_vae['model_class'] = 'vae'
            hparams_vae['n_ae_latents'] = \
                n_ae_latents + len(label_names) + hparams.get('n_background', 0)
            hparams_vae['experiment_name'] = vae_experiment_name
            hparams_vae['vae.beta'] = 1
            hparams_vae['vae.beta_anneal_epochs'] = 100
            # programmatically fill out other hparams options
            get_lab_example(hparams_vae, lab, expt)
            if 'sessions_csv' not in hparams_vae:
                hparams_vae['animal'] = animal
                hparams_vae['session'] = session
            hparams_vae['session_dir'], sess_ids = get_session_dir(hparams_vae)
            hparams_vae['expt_dir'] = get_expt_dir(hparams_vae)
            version = 0
            hparams_vae['n_sessions_per_batch'] = 1
            # use data_gen from another model so labels are loaded
            model_vae, _ = get_best_model_and_data(hparams_vae, load_data=False, version=version)
            model_vae.eval()
        elif model[:3] == 'sss':
            from behavenet.plotting.cond_ae_utils import _get_sssvae_hparams
            hparams = _get_sssvae_hparams(
                model_class='sss-vae', alpha=alpha, beta=beta, gamma=gamma,
                n_ae_latents=n_ae_latents,
                experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)
            hparams['n_ae_latents'] += len(label_names) + hparams.get('n_background', 0)
            # programmatically fill out other hparams options
            get_lab_example(hparams, lab, expt)
            if 'sessions_csv' not in hparams:
                hparams['animal'] = animal
                hparams['session'] = session
            hparams['session_dir'], sess_ids = get_session_dir(hparams)
            hparams['expt_dir'] = get_expt_dir(hparams)
            _, version = experiment_exists(hparams, which_version=True)
            hparams['n_sessions_per_batch'] = 1
            model_ae, data_gen = get_best_model_and_data(hparams, load_data=True, version=version)
            model_ae.eval()
        elif model[:2] == 'ps':
            from behavenet.plotting.cond_ae_utils import _get_psvae_hparams
            rng_seed_model = 0
            hparams = _get_psvae_hparams(
                model_class='ps-vae', alpha=alpha, beta=beta, gamma=gamma,
                n_ae_latents=n_ae_latents,
                experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)
            hparams['n_ae_latents'] += len(label_names) + hparams.get('n_background', 0)
            # programmatically fill out other hparams options
            get_lab_example(hparams, lab, expt)
            if 'sessions_csv' not in hparams:
                hparams['animal'] = animal
                hparams['session'] = session
            hparams['session_dir'], sess_ids = get_session_dir(hparams)
            hparams['expt_dir'] = get_expt_dir(hparams)
            _, version = experiment_exists(hparams, which_version=True)
            hparams['n_sessions_per_batch'] = 1
            model_ae, data_gen = get_best_model_and_data(hparams, load_data=True, version=version)
            model_ae.eval()
        elif model[:4] == 'msps':
            from behavenet.plotting.cond_ae_utils import _get_psvae_hparams
            hparams = _get_psvae_hparams(
                model_class='msps-vae', alpha=alpha, beta=beta, delta=delta,
                n_ae_latents=n_ae_latents,
                experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)
            hparams['n_ae_latents'] += len(label_names) + hparams['n_background']
            # programmatically fill out other hparams options
            get_lab_example(hparams, lab, expt)
            # hparams['animal'] = animal
            # hparams['session'] = session
            hparams['session_dir'], sess_ids = get_session_dir(hparams)
            hparams['expt_dir'] = get_expt_dir(hparams)
            _, version = experiment_exists(hparams, which_version=True)
            hparams['n_sessions_per_batch'] = 1
            model_ae, data_gen = get_best_model_and_data(hparams, load_data=True, version=version)
            model_ae.eval()
        else:
            raise Exception

        if model == 'vae':
            m, lrs_vae = compute_r2(
                'unsupervised', hparams_vae, model_vae, data_gen, version, label_names,
                dtype=dtype, overwrite=overwrite, save_results=save_results)
            metrics_df[model] = m
        elif model == 'sss-vae-u' or model == 'ps-vae-u' or model == 'msps-vae-u':
            m, lrs_sss = compute_r2(
                'unsupervised', hparams, model_ae, data_gen, version, label_names,
                dtype=dtype, overwrite=overwrite, save_results=save_results)
            metrics_df[model] = m
        elif model == 'sss-vae-s' or model == 'ps-vae-s' or model == 'msps-vae-s':
            metrics_df[model], _ = compute_r2(
                'supervised', hparams, model_ae, data_gen, version, label_names,
                dtype=dtype, overwrite=overwrite, save_results=save_results)
        elif model == 'sss-vae' or model == 'ps-vae' or model == 'msps-vae':
            metrics_df[model], lrs_sssf = compute_r2(
                'full', hparams, model_ae, data_gen, version, label_names,
                dtype=dtype, overwrite=overwrite, save_results=save_results)
        else:
            raise Exception

    # ------------------------------------------
    # collect results
    # ------------------------------------------
    trials = metrics_df[models[0]].Trial.unique()
    m0 = 'vae' # models[0] if (models[0] != 'sss-vae-s' or models[0] != 'ps-vae-s') else models[1]

    # make new dataframe that combines two outputs
    metrics_dfs = []
    for l in label_names:
        for j in trials:

            for model in models:

                if model == 'sss-vae-u':
                    model_ = 'sss-vae (unsuper. subspace)'
                elif model == 'sss-vae-s':
                    model_ = 'sss-vae (super. subspace)'
                elif model == 'ps-vae-u':
                    model_ = 'ps-vae (unsuper. subspace)'
                elif model == 'ps-vae-s':
                    model_ = 'ps-vae (super. subspace)'
                elif model == 'msps-vae-u':
                    model_ = 'msps-vae (unsuper. subspace)'
                elif model == 'msps-vae-s':
                    model_ = 'msps-vae (super. subspace)'
                else:
                    model_ = model

                df = metrics_df[model][
                    (metrics_df[model].Trial == j)
                    & (metrics_df[model].Label == l)
                    & ~(metrics_df[model].Model == 'baseline')]
                sessions = df.Session.unique()

                if len(sessions) > 1:
                    for session in sessions:
                        mse = df[df.Session == session].MSE.values[0]
                        r2 = df[df.Session == session].R2.values[0]
                        metrics_dfs.append(pd.DataFrame({
                            'Trial': j,
                            'Session': int(session),
                            'Label': l,
                            'R2': r2,
                            'MSE': mse,
                            'Model': model_}, index=[0]))
                else:
                    mse = df.MSE.values[0]
                    r2 = df.R2.values[0]
                    metrics_dfs.append(pd.DataFrame({
                        'Trial': j,
                        'Label': l,
                        'R2': r2,
                        'MSE': mse,
                        'Model': model_}, index=[0]))

                if model_ == 'vae':
                    # construct baseline once
                    df = metrics_df[m0][
                        (metrics_df[m0].Trial == j)
                        & (metrics_df[m0].Label == l)
                        & (metrics_df[m0].Model == 'baseline')]
                    if len(sessions) > 1:
                        for session in sessions:
                            mse = df[df.Session == session].MSE.values[0]
                            r2 = df[df.Session == session].R2.values[0]
                            metrics_dfs.append(pd.DataFrame({
                                'Trial': j,
                                'Session': int(session),
                                'Label': l,
                                'R2': r2,
                                'MSE': mse,
                                'Model': 'baseline'}, index=[0]))
                    else:
                        mse = df.MSE.values[0]
                        r2 = df.R2.values[0]
                        metrics_dfs.append(pd.DataFrame({
                            'Trial': j,
                            'Label': l,
                            'R2': r2,
                            'MSE': mse,
                            'Model': 'baseline'}, index=[0]))

    metrics_dfs = pd.concat(metrics_dfs)

    # ------------------------------------------
    # plot data
    # ------------------------------------------
    sns.set_style('white')
    sns.set_context('talk')

    if measure == 'r2':
        data_queried = metrics_dfs[(metrics_dfs.R2 > 0)]
        splt = sns.catplot(x='Label', y='R2', hue='Model', data=data_queried, kind='bar')
        splt.ax.set_ylabel('$R^2$')
        splt.ax.set_xlabel('Label')
    else:
        data_queried = metrics_dfs
        splt = sns.catplot(x='Label', y='MSE', hue='Model', data=data_queried, kind='bar')
        splt.ax.set_xlabel('Label')
        splt.ax.set_ylabel('MSE')
        splt.ax.set_yscale('log')

    splt.set_xticklabels(rotation=45, horizontalalignment='right')

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    return metrics_dfs


def plot_reconstruction_traces_wrapper(
        lab, expt, animal, session, alpha, beta, gamma, n_ae_latents, rng_seed_model,
        label_names, sssvae_experiment_name, vae_experiment_name, trials,
        models=['sss-vae-s', 'vae'], xtick_locs=None, frame_rate=None, scale=0.5, add_legend=True,
        save_file=None, format='pdf', **kwargs):

    if any([m.find('sss') > -1 for m in models]):
        from behavenet.plotting.cond_ae_utils import _get_sssvae_hparams
        hparams = _get_sssvae_hparams(
            model_class='sss-vae', alpha=alpha, beta=beta, gamma=gamma, n_ae_latents=n_ae_latents,
            experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)
    else:
        from behavenet.plotting.cond_ae_utils import _get_psvae_hparams
        hparams = _get_psvae_hparams(
            model_class='ps-vae', alpha = alpha, beta=beta, gamma=gamma, n_ae_latents=n_ae_latents,
            experiment_name=sssvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)

    hparams['n_ae_latents'] += len(label_names)

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    _, version_sss = experiment_exists(hparams, which_version=True)
    model_sss, data_gen = get_best_model_and_data(
        hparams, Model=None, load_data=True, version=version_sss)
    model_sss.eval()

    for model in models:
        print()
        print(model)
        if model == 'vae':
            hparams_vae = copy.deepcopy(hparams)
            hparams_vae['model_class'] = 'vae'
            hparams_vae['n_ae_latents'] = n_ae_latents + len(label_names)
            hparams_vae['experiment_name'] = vae_experiment_name
            hparams_vae['vae.beta'] = 1
            hparams_vae['vae.beta_anneal_epochs'] = 100
            hparams_vae['expt_dir'] = get_expt_dir(hparams_vae)
            version = 0
            model_vae, _ = get_best_model_and_data(
                hparams_vae, Model=None, load_data=False, version=version)
            model_vae.eval()
            _, lrs_vae = compute_r2(
                'unsupervised', hparams_vae, model_vae, data_gen, version, label_names)
        elif model == 'sss-vae-u' or model == 'ps-vae-u':
            _, lrs_sss = compute_r2(
                'unsupervised', hparams, model_sss, data_gen, version_sss, label_names)
        elif model == 'sss-vae-s' or model == 'ps-vae-s':
            # use sss-vae model instead of post-hoc regression model
            pass
        elif model == 'sss-vae' or model == 'ps-vae':
            _, lrs_sssf = compute_r2(
                'full', hparams, model_sss, data_gen, version_sss, label_names)
        else:
            raise Exception

    # loop over trials to plot
    for trial in trials:

        batch = data_gen.datasets[0][trial]
        labels_og = batch['labels'].detach().cpu().numpy()  # [:, 2:]
        labels_pred_sss_vae = model_sss.get_predicted_labels(
            batch['images'].to(hparams['device'])).detach().cpu().numpy()
        if 'labels_masks' in batch:
            labels_masks = batch['labels_masks'].detach().cpu().numpy()
            labels_og[labels_masks == 0] = np.nan

        if save_file is not None:
            save_file_trial = save_file + '_trial-%i' % trial
        else:
            save_file_trial = None

        if 'vae' in models and 'sss-vae-u' in models and 'sss-vae-s' in models:
            # vae
            if hparams_vae['model_class'] == 'ae':
                latents, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
            else:
                latents, _, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
            labels_pred_vae = get_predicted_labels(lrs_vae, latents.detach().cpu().numpy())
            # sss-vae-s
            _, latents, _, _, _ = model_sss.encoding(batch['images'].to(hparams['device']))
            labels_pred_sss_vae_s = get_predicted_labels(lrs_sss, latents.detach().cpu().numpy())

            plot = plot_reconstruction_traces(
                [labels_og, labels_pred_vae, labels_pred_sss_vae, labels_pred_sss_vae_s],
                ['original', 'vae', 'sss-vae (super)', 'sss-vae (unsuper)'],
                scale=scale, xtick_locs=xtick_locs, frame_rate=frame_rate, add_legend=add_legend,
                save_file=save_file_trial, format=format)
        #         plot = plot_reconstruction_traces(
        #             [labels_og, labels_pred_vae, labels_pred_sss_vae],
        #             ['original', 'vae', 'sss-vae (super)'],
        #             scale=0.25)

        elif 'vae' in models:
            if hparams_vae['model_class'] == 'ae':
                latents, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
            else:
                latents, _, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
            labels_pred_vae = get_predicted_labels(lrs_vae, latents.detach().cpu().numpy())
            #         plot = plot_reconstruction_traces(
            #             [labels_og, labels_pred_vae, labels_pred_sss_vae],
            #             ['original', 'vae', 'sss-vae'],
            #             scale=0.25)
            plot = plot_reconstruction_traces(
                [labels_og, labels_pred_sss_vae, labels_pred_vae],
                ['original', 'ps-vae', 'vae'],
                scale=scale, add_legend=add_legend, xtick_locs=xtick_locs, frame_rate=frame_rate,
                save_file=save_file_trial, format=format)

        elif 'sss-vae' in models or 'ps-vae' in models:
            if 'sss-vae' in models:
                titles = ['original', 'sss-vae-full', 'sss-vae-s']
            else:
                titles = ['original', 'ps-vae-full', 'ps-vae-s']
            y, w, _, _, _ = model_sss.encoding(batch['images'].to(hparams['device']))
            latents = np.hstack([y.detach().cpu().numpy(), w.detach().cpu().numpy()])
            labels_pred_full = get_predicted_labels(lrs_sssf, latents)
            plot = plot_reconstruction_traces(
                [labels_og, labels_pred_full, labels_pred_sss_vae],
                titles,
                scale=scale, add_legend=add_legend, xtick_locs=xtick_locs, frame_rate=frame_rate,
                save_file=save_file_trial, format=format)
        else:  # compare supervised and unsupervised trace reconstructions
            if 'sss-vae-u' in models:
                titles = ['original', 'sss-vae-u', 'sss-vae-s']
            else:
                titles = ['original', 'ps-vae-u', 'ps-vae-s']
            _, latents, _, _, _ = model_sss.encoding(batch['images'].to(hparams['device']))
            labels_pred = get_predicted_labels(lrs_sss, latents.detach().cpu().numpy())
            plot = plot_reconstruction_traces(
                [labels_og, labels_pred, labels_pred_sss_vae],
                titles,
                scale=scale, add_legend=add_legend, xtick_locs=xtick_locs, frame_rate=frame_rate,
                save_file=save_file_trial, format=format)


def plot_msps_reconstruction_traces_wrapper(
        lab, expt, alpha, beta, delta, n_ae_latents, rng_seed_model,
        label_names, mspsvae_experiment_name, vae_experiment_name, trials, sess_idxs,
        models=['msps-vae-s', 'vae'], xtick_locs=None, frame_rate=None, scale=0.5, add_legend=True,
        save_file=None, format='pdf', **kwargs):

    from behavenet.plotting.cond_ae_utils import _get_psvae_hparams
    hparams = _get_psvae_hparams(
        model_class='msps-vae', alpha=alpha, beta=beta, delta=delta, n_ae_latents=n_ae_latents,
        experiment_name=mspsvae_experiment_name, rng_seed_model=rng_seed_model, **kwargs)

    hparams['n_ae_latents'] += len(label_names) + hparams['n_background']

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    _, version = experiment_exists(hparams, which_version=True)
    hparams['n_sessions_per_batch'] = 1
    model_msps, data_gen = get_best_model_and_data(hparams, load_data=True, version=version)
    model_msps.eval()

    for model in models:
        print()
        print(model)
        if model == 'vae':
            hparams_vae = copy.deepcopy(hparams)
            hparams_vae['model_class'] = 'vae'
            hparams_vae['n_ae_latents'] = n_ae_latents + len(label_names) + hparams['n_background']
            hparams_vae['experiment_name'] = vae_experiment_name
            hparams_vae['vae.beta'] = 1
            hparams_vae['vae.beta_anneal_epochs'] = 100
            hparams_vae['expt_dir'] = get_expt_dir(hparams_vae)
            version = 0
            model_vae, _ = get_best_model_and_data(hparams_vae, load_data=False, version=version)
            model_vae.eval()
            _, lrs_vae = compute_r2(
                'unsupervised', hparams_vae, model_vae, data_gen, version, label_names)
        elif model == 'msps-vae-u':
            _, lrs_msps = compute_r2(
                'unsupervised', hparams, model_ae, data_gen, version, label_names)
        elif model == 'msps-vae-s':
            # use sss-vae model instead of post-hoc regression model
            pass
        elif model == 'msps-vae':
            _, lrs_mspsf = compute_r2(
                'full', hparams, model_ae, data_gen, version, label_names)
        else:
            raise Exception

    # loop over trials to plot
    for sess_idx in sess_idxs:

        if save_file is not None:
            save_file_sess = save_file + '_sess-%i' % sess_idx

        for trial in trials:

            batch = data_gen.datasets[sess_idx][trial]
            labels_og = batch['labels'].detach().cpu().numpy()  # [:, 2:]
            labels_pred_msps_vae = model_msps.get_predicted_labels(
                batch['images'].to(hparams['device'])).detach().cpu().numpy()
            if 'labels_masks' in batch:
                labels_masks = batch['labels_masks'].detach().cpu().numpy()
                labels_og[labels_masks == 0] = np.nan

            if save_file is not None:
                save_file_trial = save_file_sess + '_trial-%i' % trial
            else:
                save_file_trial = None

            if 'vae' in models:
                if hparams_vae['model_class'] == 'ae':
                    latents, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
                else:
                    latents, _, _, _ = model_vae.encoding(batch['images'].to(hparams['device']))
                labels_pred_vae = get_predicted_labels(lrs_vae, latents.detach().cpu().numpy())

                plot = plot_reconstruction_traces(
                    [labels_og, labels_pred_msps_vae, labels_pred_vae],
                    ['original', 'msps-vae', 'vae'],
                    scale=scale, add_legend=add_legend, xtick_locs=xtick_locs,
                    frame_rate=frame_rate, save_file=save_file_trial, format=format)

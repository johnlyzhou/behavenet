import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pickle
import seaborn as sns

from behavenet import make_dir_if_not_exists
from behavenet.fitting.utils import get_expt_dir
from behavenet.plotting import save_movie
from psvae_utils.ssmutils import fit_with_random_restarts


def load_latents(hparams, version, n_labels=None):

    # load latents
    hparams['expt_dir'] = get_expt_dir(hparams)
    sess_id = str('%s_%s_%s_%s_latents.pkl' % (
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))
    filename = os.path.join(
        hparams['expt_dir'], 'version_%i' % version, sess_id)
    if not os.path.exists(filename):
        raise NotImplementedError
    latent_dict = pickle.load(open(filename, 'rb'))
    print('loaded latents from %s' % filename)

    dtypes = ['train', 'val', 'test']
    latents = {dtype: [] for dtype in dtypes}
    latents_all = {dtype: [] for dtype in dtypes}
    dtype_idxs = {dtype: [] for dtype in dtypes}
    for dtype in dtypes:
        for trial in latent_dict['trials'][dtype]:
            if hparams['model_class'] == 'sss-vae':
                if dtype == 'test':
                    # combine val/test data
                    latents['val'].append(latent_dict['latents'][trial][:, n_labels:])
                    latents_all['val'].append(latent_dict['latents'][trial])
                    dtype_idxs['val'].append(trial)
                else:
                    latents[dtype].append(latent_dict['latents'][trial][:, n_labels:])
                    latents_all[dtype].append(latent_dict['latents'][trial])
                    dtype_idxs[dtype].append(trial)
            else:
                if dtype == 'test':
                    # combine val/test data
                    latents['val'].append(latent_dict['latents'][trial])
                    dtype_idxs['val'].append(trial)
                else:
                    latents[dtype].append(latent_dict['latents'][trial])
                    dtype_idxs[dtype].append(trial)

    return latents, dtype_idxs, latents_all


def fit_arhmm(
        hparams, version, data, K=2, lags=1, obs='ar', num_restarts=5, num_iters=150,
        init_type='random', dir_string=''):
    save_path = os.path.join(hparams['session_dir'], 'arhmm', '%02i-%s-version-%i_%s-init%s' % (
        hparams['n_ae_latents'], hparams['model_class'], version, init_type, dir_string))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model, lps, models_all, lps_all = fit_with_random_restarts(
        K, data[0].shape[1], obs, lags, data,
        num_restarts=num_restarts, num_iters=num_iters,
        method='em', save_path=save_path, init_type=init_type)
    return model, lps, models_all, lps_all


def plot_arhmm_training_curves(lps, plot_all_restarts=True):
    plt.figure()
    if plot_all_restarts:
        for j, restart in enumerate(lps):
            plt.plot(restart, 'k')
    else:
        plt.plot(lps, 'k', color='k')
    plt.xlabel('Epoch')
    plt.ylabel('Log probability')
    # plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()


def confusion_matrix(true_states, inf_states, num_states):
    confusion = np.zeros((num_states, num_states))
    ztotal = np.zeros((num_states, 1))
    for i in range(num_states):
        for ztrue, zinf in zip(true_states, inf_states):
            for j in range(num_states):
                confusion[i, j] += np.sum((ztrue == i) & (zinf == j))
            ztotal[i] += np.sum(ztrue == i)
    return confusion / ztotal


def extract_snippets(
        states, latents, align_idxs, t_back=5, t_forward=10, still_state=0, include_all=False):
    latent_snippets = []
    switch_idxs = []
    for state, latent, align_idx in zip(states, latents, align_idxs):
        if align_idx < 0:
            if include_all:
                latent_snippets.append([])
                switch_idxs.append(np.nan)
            continue
        # find state switch
        if state[align_idx] == still_state:
            # move forward
            j = align_idx
            while state[j] == still_state:
                j += 1
                if j >= len(state):
                    break
        else:
            # move back
            j = align_idx
            while state[j] != still_state:
                j -= 1
                if j < 0:
                    break
            j += 1
        if j - t_back < 0 or j + t_forward > len(state):
            if include_all:
                latent_snippets.append([])
                switch_idxs.append(np.nan)
            continue
        latent_snippets.append(latent[j - t_back:j + t_forward])
        switch_idxs.append(j)
    return latent_snippets, switch_idxs


def extract_snippets_state_change(states, latents, t_back=5, t_forward=5):
    from behavenet.plotting.arhmm_utils import get_discrete_chunks
    chunks = get_discrete_chunks(states)
    snippets = []
    for chunk in chunks[1]:  # 0 is set to "still" state
        tr = chunk[0]
        idx_beg = chunk[1]
        idx_end = chunk[2]
        if idx_beg - t_back < 0:
            continue
        if idx_beg + t_forward > latents[tr].shape[0]:
            continue
        snippets.append(latents[tr][idx_beg - t_back:idx_beg + t_forward, :])
    return snippets


def plot_snippets(
        latent_snippets, t_back, frame_rate=30, alpha=0.1, figsize=(6, 4), add_mean=True, scale=2,
        colored_traces=False, mean_linestyle='-r', save_file=None, format='pdf'):
    fig = plt.figure(figsize=figsize)

    spc = scale * abs(np.percentile(np.concatenate(latent_snippets), 98))
    n_latents = latent_snippets[0].shape[1]

    xs = (np.arange(latent_snippets[0].shape[0]) - t_back + 1) / frame_rate

    # plot individual trials
    for snippet in latent_snippets:
        plotting_snippet = snippet + spc * np.arange(n_latents)
        if colored_traces:
            plt.gca().set_prop_cycle(None)
            plt.plot(xs, plotting_snippet, '-', lw=1, alpha=alpha)
        else:
            plt.plot(xs, plotting_snippet, '-k', lw=1, alpha=alpha)

    # plot averages
    if add_mean:
        latent_avg = np.mean(np.concatenate(
            [l[:, :, None] for l in latent_snippets], axis=2), axis=2)
        plotting_snippet = latent_avg + spc * np.arange(n_latents)
        plt.plot(xs, plotting_snippet, mean_linestyle, lw=2)

    plt.ylim([-spc, n_latents * spc])
    plt.axvline(x=0, color='k', linestyle='--')
    plt.yticks([])
    plt.xlabel('Switch from still to move')

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    plt.show()


def concatenate_video_clips(
        ims, save_file=None, max_frames=400, frame_rate=10, n_buffer=5, n_pre_frames=5,
        fig_width=5, text_color=[1, 1, 1]):
    """Sequentially present video clips with optional state change indicator box

    Parameters
    ----------
    ims : :obj:`list`
        each entry is a numpy array of shape (n_frames, y_pix, x_pix)
    save_file : :obj:`str`
        full save file (path and filename)
    max_frames : :obj:`int`, optional
        maximum number of frames to animate
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    n_buffer : :obj:`int`, optional
        number of blank frames between clips
    n_pre_frames : :obj:`int`, optional
        number of behavioral frames to precede an indicator box
    fig_width : :obj:`float`, optional
        width of figure in inches

    """

    # get video dims
    bs, y_dim, x_dim = ims[0].shape

    fig_height = y_dim / x_dim * fig_width
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    ax.set_yticks([])
    ax.set_xticks([])

    imshow_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    txt_kwargs = {
        'fontsize': 16, 'color': text_color, 'horizontalalignment': 'left',
        'verticalalignment': 'center', 'transform': ax.transAxes}

    ims_list = []
    im_counter = 0

    for i, curr_ims in enumerate(ims):

        # Loop over this chunk
        for j, curr_im in enumerate(curr_ims):

            ims_list_curr = []

            # add frame
            im = ax.imshow(curr_im, **imshow_kwargs)
            ims_list_curr.append(im)

            # add text
            if j < n_pre_frames:
                im = ax.text(0.02, 0.06, 'Rest', **txt_kwargs)
            else:
                im = ax.text(0.02, 0.06, 'Movement', **txt_kwargs)
            ims_list_curr.append(im)

            # add box
            #             if n_pre_frames is not None and (n_pre_frames <= j < (n_pre_frames + 2)):
            #                 rect = matplotlib.patches.Rectangle(
            #                     (5, 5), 10, 10, linewidth=1, edgecolor='r', facecolor='r')
            #                 im = ax.add_patch(rect)
            #                 ims_list_curr.append(im)

            ims_list.append(ims_list_curr)
            im_counter += 1

        # Add buffer black frames
        for j in range(n_buffer):
            im = ax.imshow(np.zeros((y_dim, x_dim)), **imshow_kwargs)
            ims_list.append([im])
            im_counter += 1

        # break if we've exceed max frames
        if im_counter > max_frames:
            break

    #     plt.tight_layout(pad=0)

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(fig, ims_list, interval=20, blit=True, repeat=False)
    print('done')

    save_movie(save_file, ani, frame_rate=frame_rate)


def concatenate_video_clips_wtraces(
        ims, latents, save_file=None, max_frames=400, frame_rate=10, n_buffer=5, n_pre_frames=5,
        fig_width=5, scale=2, text_color=[1, 1, 1], colors=None, labels=None, alpha=0.2):
    """Sequentially present video clips with optional state change indicator box

    Parameters
    ----------
    ims : :obj:`list`
        each entry is a numpy array of shape (n_frames, y_pix, x_pix)
    latents : :obj:`list`
        each entry is a numpy array of shape (n_frames, n_latents)
    save_file : :obj:`str`
        full save file (path and filename)
    max_frames : :obj:`int`, optional
        maximum number of frames to animate
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    n_buffer : :obj:`int`, optional
        number of blank frames between clips
    n_pre_frames : :obj:`int`, optional
        number of behavioral frames to precede an indicator box
    fig_width : :obj:`float`, optional
        width of figure in inches
    colors : :obj:`list`, optional
        colors for latents
    labels : :obj:`list`, optional
        labels for latents
    alpha : :obj:`float`, optional
        alpha value of latent traces

    """

    # get video dims
    bs, y_dim, x_dim = ims[0].shape
    time = np.arange(bs)

    fig_height = y_dim / x_dim * fig_width
    #     fig, axs = plt.subplots(2, 1, figsize=(fig_width, 2 * fig_height))
    # fig = plt.figure(figsize=(fig_width, 2.5 * fig_height))
    fig = plt.figure(figsize=(2 * fig_width, 2.5 * fig_height))
    fig.patch.set_facecolor('k')
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]

    # plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)
    plt.subplots_adjust(wspace=0, hspace=0, left=0.3, bottom=0, right=0.7, top=1)

    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])

    imshow_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    txt_kwargs = {
        'fontsize': 16, 'color': text_color, 'horizontalalignment': 'left',
        'verticalalignment': 'center'}

    # plot snippets
    latents_all = np.concatenate(latents)
    std = np.std(latents_all, axis=0)
    latents_sc = [l / std for l in latents]
    spc = scale * abs(np.percentile(np.concatenate(latents_sc), 98))
    n_latents = latents[0].shape[1]
    colors = sns.color_palette() if colors is None else colors
    for l in latents_sc:
        for k in range(n_latents):
            axs[1].plot(l[:, k] + spc * k, '-', color=colors[k], lw=1, alpha=alpha)
    if labels is not None:
        for k in range(n_latents):
            y_offset = (k + 1) / (n_latents + 1) + 0.1
            axs[1].text(
                0.05, y_offset, labels[k], fontsize=16,
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[1].set_ylim([-spc, n_latents * spc])
    axs[1].axvline(x=n_pre_frames - 1, color='k', linestyle='--')
    axs[1].set_yticks([])

    ims_list = []
    im_counter = 0

    for i, curr_ims in enumerate(ims):

        # Loop over this chunk
        for j, curr_im in enumerate(curr_ims):

            ims_list_curr = []

            # add frame
            im = axs[0].imshow(curr_im, **imshow_kwargs)
            ims_list_curr.append(im)

            # add text
            if j < n_pre_frames:
                im = axs[0].text(
                    0.02, 0.06, 'Rest', transform=axs[0].transAxes, **txt_kwargs)
            else:
                im = axs[0].text(
                    0.02, 0.06, 'Movement', transform=axs[0].transAxes, **txt_kwargs)
            ims_list_curr.append(im)

            # add latents
            for k in range(n_latents):
                im = axs[1].plot(
                    time[:j + 1], latents_sc[i][:j + 1, k] + spc * k,
                    '-', color=colors[k], lw=2, alpha=1)[0]
                ims_list_curr.append(im)

            ims_list.append(ims_list_curr)
            im_counter += 1

        # Add buffer black frames
        for j in range(n_buffer):
            im = axs[0].imshow(np.zeros((y_dim, x_dim)), **imshow_kwargs)
            ims_list.append([im])
            im_counter += 1

        # break if we've exceed max frames
        if im_counter > max_frames:
            break

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(fig, ims_list, interval=20, blit=True, repeat=False)
    print('done')

    save_movie(save_file, ani, frame_rate=frame_rate)


def plot_states_overlaid_with_latents(
        latents, states, save_file=None, ax=None, xtick_locs=None, frame_rate=None, cmap='tab20b',
        format='png', colored_traces=False, color_cyle=None):
    """Plot states for a single trial overlaid with latents.

    Parameters
    ----------
    latents : :obj:`np.ndarray`
        shape (n_frames, n_latents)
    states : :obj:`np.ndarray`
        shape (n_frames,)
    save_file : :obj:`str`, optional
        full save file (path and filename)
    ax : :obj:`matplotlib.Axes` object or :obj:`NoneType`, optional
        axes to plot into; if :obj:`NoneType`, a new figure is created
    xtick_locs : :obj:`array-like`, optional
        tick locations in bin values for plot
    frame_rate : :obj:`float`, optional
        behavioral video framerate; to properly relabel xticks
    cmap : :obj:`str`, optional
        matplotlib colormap
    format : :obj:`str`, optional
        any accepted matplotlib save format, e.g. 'png' | 'pdf' | 'jpeg'

    Returns
    -------
    :obj:`matplotlib.figure.Figure`
        matplotlib figure handle if :obj:`ax=None`, otherwise updated axis

    """
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.gca()
    else:
        fig = None
    spc = 1.1 * abs(latents.max())
    n_latents = latents.shape[1]
    plotting_latents = latents + spc * np.arange(n_latents)
    ymin = min(-spc, np.min(plotting_latents))
    ymax = max(spc * n_latents, np.max(plotting_latents))
    ax.imshow(
        states[None, :], aspect='auto', extent=(0, len(latents), ymin, ymax), cmap=cmap,
        alpha=1.0)
    if colored_traces:
        if color_cyle is None:
            ax.set_prop_cycle(None)
        else:
            ax.set_prop_cycle(color=color_cyle)
        ax.plot(plotting_latents, lw=3)
    else:
        ax.plot(plotting_latents, '-k', lw=3)
    ax.set_ylim([ymin, ymax])
    #     yticks = spc * np.arange(n_latents)
    #     ax.set_yticks(yticks[::2])
    #     ax.set_yticklabels(np.arange(n_latents)[::2])
    ax.set_yticks([])
    #     ax.set_ylabel('Latent')

    ax.set_xlabel('Time (bins)')
    if xtick_locs is not None:
        ax.set_xticks(xtick_locs)
        if frame_rate is not None:
            ax.set_xticklabels((np.asarray(xtick_locs) / frame_rate).astype('int'))
            ax.set_xlabel('Time (sec)')

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        plt.savefig(save_file + '.' + format, dpi=300, format=format)

    if fig is None:
        return ax
    else:
        return fig

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import from_numpy

from behavenet import get_user_dir
from behavenet.fitting.eval import get_reconstruction
from behavenet.fitting.utils import get_best_model_and_data
from behavenet.fitting.utils import get_expt_dir
from behavenet.fitting.utils import get_session_dir
from behavenet.fitting.utils import get_lab_example
from behavenet.plotting import concat, save_movie
from behavenet.plotting.cond_ae_utils import get_crop


# --------------------------------------
# reconstruction movies
# --------------------------------------

def make_reconstruction_movie_wrapper(
        hparams, save_file, model_info, trial_idxs=None, trials=None, sess_idx=0,
        max_frames=400, xtick_locs=None, add_traces=False, label_names=None, frame_rate=15,
        layout_pattern=None):
    """Produce movie with original video, reconstructed video, and residual, with optional traces.

    This is a high-level function that loads the model described in the hparams dictionary and
    produces the necessary predicted video frames.

    Parameters
    ----------
    hparams : :obj:`dict`
        needs to contain enough information to specify an autoencoder
    save_file : :obj:`str`
        full save file (path and filename)
    trial_idxs : :obj:`list`, optional
        list of test trials to construct videos from; if :obj:`NoneType`, use first test
        trial
    sess_idx : :obj:`int`, optional
        session index into data generator
    add_traces : :obj:`bool`, optional
        add traces alongside reconstructions
    label_names : :obj:`list`, optional
        ps-vae label names
    max_frames : :obj:`int`, optional
        maximum number of frames to animate from a trial
    frame_rate : :obj:`float`, optional
        frame rate of saved movie
    layout_pattern : :obj:`np.ndarray`
        boolean array that determines where reconstructed frames are placed in a grid

    """

    from behavenet.fitting.eval import get_reconstruction
    from behavenet.fitting.utils import get_best_model_and_data
    from behavenet.plotting.ae_utils import make_reconstruction_movie
    from behavenet.plotting.cond_ae_utils import get_model_input

    n_latents = hparams['n_ae_latents']
    n_labels = hparams['n_labels']
    expt_name = hparams.get('experiment_name', None)

    # set up models to fit
    titles = ['Original']
    for model in model_info:
        titles.append(model.get('title', ''))

    # insert original video at front
    model_info.insert(0, {'model_class': None})

    ims_recon = [[] for _ in titles]
    latents = [[] for _ in titles]

    if trial_idxs is None:
        trial_idxs = [None] * len(trials)
    if trials is None:
        trials = [None] * len(trial_idxs)
    if isinstance(sess_idx, int):
        sess_idx = sess_idx * np.ones((len(trials),))

    for i, model in enumerate(model_info):

        if i == 0:
            continue

        # further specify model
        version = model.get('version', 'best')
        hparams['experiment_name'] = model.get('experiment_name', expt_name)
        hparams['model_class'] = model.get('model_class')
        model_ae, data_generator = get_best_model_and_data(hparams, version=version)

        # get images
        for trial_idx, trial, s_idx in zip(trial_idxs, trials, sess_idx):

            # get model inputs
            ims_orig_pt, ims_orig_np, _, labels_pt, labels_np, labels_2d_pt, _ = get_model_input(
                data_generator, hparams, model_ae, trial_idx=trial_idx, trial=trial,
                sess_idx=s_idx, max_frames=max_frames, compute_latents=False,
                compute_2d_labels=False)

            # get model outputs
            if hparams['model_class'] == 'labels-images':
                ims_recon_tmp = get_reconstruction(model_ae, labels_pt)
                latents_tmp = labels_np
            else:
                ims_recon_tmp, latents_tmp = get_reconstruction(
                    model_ae, ims_orig_pt, labels=labels_pt, labels_2d=labels_2d_pt,
                    return_latents=True, dataset=s_idx)
                # orient to match labels
                if hparams['model_class'] == 'ps-vae' or hparams['model_class'] == 'msps-vae':
                    latents_tmp[:, :n_labels] *= \
                        np.sign(model_ae.encoding.D.weight.data.cpu().detach().numpy())

            ims_recon[i].append(ims_recon_tmp)
            latents[i].append(latents_tmp)

            # add a couple black frames to separate trials
            final_trial = True
            if (trial_idx is not None and (trial_idx != trial_idxs[-1])) or \
                    (trial is not None and (trial != trials[-1])):
                final_trial = False

            n_buffer = 5
            if not final_trial:
                _, n, y_p, x_p = ims_recon[i][-1].shape
                ims_recon[i].append(np.zeros((n_buffer, n, y_p, x_p)))
                latents[i].append(np.nan * np.zeros((n_buffer, latents[i][-1].shape[1])))

            if i == 1:  # deal with original frames only once
                ims_recon[0].append(ims_orig_np)
                latents[0].append([])
                # add a couple black frames to separate trials
                if not final_trial:
                    _, n, y_p, x_p = ims_recon[0][-1].shape
                    ims_recon[0].append(np.zeros((n_buffer, n, y_p, x_p)))

    for i, (ims, zs) in enumerate(zip(ims_recon, latents)):
        ims_recon[i] = np.concatenate(ims, axis=0)
        latents[i] = np.concatenate(zs, axis=0)

    if layout_pattern is None:
        if len(titles) < 4:
            n_rows, n_cols = 1, len(titles)
        elif len(titles) == 4:
            n_rows, n_cols = 2, 2
        elif len(titles) > 4:
            n_rows, n_cols = 2, 3
        else:
            raise ValueError('too many models')
    else:
        assert np.sum(layout_pattern) == len(ims_recon)
        n_rows, n_cols = layout_pattern.shape
        count = 0
        for pos_r in layout_pattern:
            for pos_c in pos_r:
                if not pos_c:
                    ims_recon.insert(count, [])
                    titles.insert(count, [])
                count += 1

    if add_traces:
        make_reconstruction_movie_wtraces(
            ims=ims_recon, latents=latents, titles=titles, xtick_locs=xtick_locs,
            frame_rate_beh=hparams['frame_rate'], scale=0.3, label_names=label_names,
            save_file=save_file, frame_rate=frame_rate)
    else:
        make_reconstruction_movie(
            ims=ims_recon, titles=titles, n_rows=n_rows, n_cols=n_cols,
            save_file=save_file, frame_rate=frame_rate)


def make_reconstruction_movie_wtraces(
        ims, latents, titles, xtick_locs, frame_rate_beh, scale=0.5, label_names=None,
        save_file=None, frame_rate=None, show_residuals=True):
    """Inflexible function for plotting recons, residuals, and latents for several models."""

    from matplotlib.gridspec import GridSpec

    n_channels, y_pix, x_pix = ims[0].shape[1:]
    n_time, n_ae_latents = latents[1].shape

    n_rows = len(ims)
    if show_residuals:
        n_cols = 3
        fig_width = 12
        width_ratios = [1, 1, 2]
        im_cols = 2
    else:
        n_cols = 2
        fig_width = 9
        width_ratios = [1, 2]
        im_cols = 1

    fig_height = 3 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios)
    axs = []
    for c in range(n_cols):
        for r in range(n_rows):
            # col 0, then col 1
            axs.append(fig.add_subplot(gs[r, c]))
    for i, ax in enumerate(axs):
        ax.set_yticks([])
        if i == len(ims) or i == len(ims) * im_cols:
            ax.set_axis_off()  # assume original frames first, no latents
        if i > len(ims) * im_cols:
            ax.get_xaxis().set_tick_params(labelsize=12, direction='in')
        elif i < len(ims) * im_cols:
            ax.set_xticks([])
        else:
            ax.set_axis_off()  # assume original frames first, no latents

    # check that the axes are correct
    fontsize = 12
    idx = 0
    for title in titles:
        axs[idx].set_title(titles[idx], fontsize=fontsize)
        idx += 1
    # blank (legend)
    idx += 1
    if show_residuals:
        for t in range(len(ims) - 1):
            axs[idx].set_title('Residual')
            idx += 1
        # blank
        idx += 1
    # ps-vae latents
    axs[idx].set_title('MSPS-VAE latents', fontsize=fontsize)
    axs[idx].set_xticklabels([])
    if xtick_locs is not None and frame_rate_beh is not None:
        axs[idx].set_xticks(xtick_locs)
    idx += 1
    axs[idx].set_title('VAE latents', fontsize=fontsize)
    if len(ims) > 3:
        # take care of VAE ticks
        axs[idx].set_xticklabels([])
        if xtick_locs is not None and frame_rate_beh is not None:
            axs[idx].set_xticks(xtick_locs)
        # labels-images
        idx += 1
        axs[idx].set_title('Labels', fontsize=fontsize)
    # properly label x-axis of final row
    if xtick_locs is not None and frame_rate_beh is not None:
        axs[idx].set_xticks(xtick_locs)
        axs[idx].set_xticklabels((np.asarray(xtick_locs) / frame_rate_beh).astype('int'))
        axs[idx].set_xlabel('Time (s)', fontsize=fontsize)
    else:
        axs[idx].set_xlabel('Time (bins)', fontsize=fontsize)

    im_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    tr_kwargs = {'animated': True, 'linewidth': 2}
    txt_kwargs = {
        'fontsize': 10, 'horizontalalignment': 'left', 'verticalalignment': 'center'}

    latents_ae_color = [0.2, 0.2, 0.2]

    # -----------------
    # labels
    # -----------------
    if label_names is not None:
        idx = len(ims)
        axs[idx].set_prop_cycle(None)  # reset colors
        for l, label in enumerate(label_names):
            c = axs[idx]._get_lines.get_next_color()
            y_val = l / (len(label_names) + 2) + 1 / (len(label_names) + 2)
            axs[idx].plot(
                [0.1, 0.15], [y_val, y_val], '-', color=c, transform=axs[idx].transAxes)
            axs[idx].text(
                0.17, y_val, label, color='k', transform=axs[idx].transAxes, **txt_kwargs)

    time = np.arange(n_time)

    # normalize traces
    latents_sc = []
    for zs in latents:
        if len(zs) == 0:
            latents_sc.append(None)
        else:
            means = np.nanmean(zs, axis=0)
            stds = np.nanstd(zs, axis=0) / scale
            latents_sc.append((zs - means) / stds)

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims_ani = []
    for i in range(n_time):

        ims_curr = []
        idx = 0

        if i % 100 == 0:
            print('processing frame %03i/%03i' % (i, n_time))

        # -----------------
        # behavioral videos
        # -----------------
        for idx in range(n_rows):
            ims_tmp = ims[idx][i, 0] if n_channels == 1 else concat(ims[idx][i])
            im = axs[idx].imshow(ims_tmp, **im_kwargs)
            ims_curr.append(im)

        # -----------------
        # residuals
        # -----------------
        if show_residuals:
            for idx in range(1, n_rows):
                ims_tmp = ims[idx][i, 0] if n_channels == 1 else concat(ims[idx][i])
                ims_og = ims[0][i, 0] if n_channels == 1 else concat(ims[0][i])
                im = axs[n_rows + idx].imshow(ims_tmp - ims_og + 0.5, **im_kwargs)
                ims_curr.append(im)

        # -----------------
        # traces
        # -----------------
        # latents over time
        for idx in range(n_rows * im_cols + 1, n_rows * im_cols + n_rows):
            axs[idx].set_prop_cycle(None)  # reset colors
            for latent in range(latents_sc[idx - n_rows * im_cols].shape[1]):
                if idx == n_rows * im_cols + 1:
                    latents_color = axs[idx]._get_lines.get_next_color()
                elif idx == n_rows * im_cols + 3:
                    # hack to get labels-images traces w/ colors
                    latents_color = axs[idx]._get_lines.get_next_color()
                else:
                    latents_color = [0, 0, 0]
                im = axs[idx].plot(
                    time[0:i + 1], latent + latents_sc[idx - n_rows * im_cols][0:i + 1, latent],
                    color=latents_color, alpha=0.7, **tr_kwargs)[0]
                axs[idx].spines['top'].set_visible(False)
                axs[idx].spines['right'].set_visible(False)
                axs[idx].spines['left'].set_visible(False)
                ims_curr.append(im)

        ims_ani.append(ims_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims_ani, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


# --------------------------------------
# latent/label traversal functions
# --------------------------------------

def plot_frame_array_labels(
        hparams, ims, plot_func, interp_func, crop_type, markers, save_outputs=False, **kwargs):

    n_frames = len(ims[0])

    marker_kwargs = {
        'markersize': 20, 'markeredgewidth': 5, 'markeredgecolor': [1, 1, 0],
        'fillstyle': 'none'}

    if save_outputs:
        save_file = os.path.join(
            get_user_dir('fig'),
            'ae', 'D=%02i_label-manipulation_%s_%s-crop.png' %
                  (hparams['n_ae_latents'], hparams['session'], crop_type))
    else:
        save_file = None

    if plot_func.__name__ == 'plot_2d_frame_array':
        """plot generated frames and differences separately"""

        # plot generated frames
        if crop_type:
            plot_func(
                ims, markers=None, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)
        else:
            plot_func(
                ims, markers=markers, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)

        # plot differences
        if interp_func.__name__ == 'interpolate_2d':
            # use upper left corner as base frame for whole grid
            base_im = ims[0][0]
            ims_diff = [[None for _ in range(n_frames)] for _ in range(n_frames)]
            for r, ims_list_y in enumerate(ims):
                for c, im in enumerate(ims_list_y):
                    ims_diff[r][c] = 0.5 + (im - base_im)
        else:
            # use left-most column as base frame for each row
            ims_diff = [[None for _ in range(n_frames)] for _ in range(len(ims))]
            for r, ims_list_y in enumerate(ims):
                for c, im in enumerate(ims_list_y):
                    ims_diff[r][c] = 0.5 + (im - ims[r][0])  # compare across rows

        plot_func(
            ims_diff, markers=markers, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)

    else:
        """plot generated frames and differences together"""
        if crop_type:
            plot_func(
                ims, markers=None, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)
        else:
            plot_func(
                ims, markers=None, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)


def plot_frame_array_latents(
        hparams, ims, plot_func, interp_func, n_latents, crop_type, markers, save_outputs=False,
        **kwargs):

    n_frames = len(ims[0])

    if crop_type:
        marker_kwargs = {
            'markersize': 30, 'markeredgewidth': 8, 'markeredgecolor': [1, 1, 0],
            'fillstyle': 'none'}
    else:
        marker_kwargs = {
            'markersize': 20, 'markeredgewidth': 5, 'markeredgecolor': [1, 1, 0],
            'fillstyle': 'none'}

    if save_outputs:
        save_file = os.path.join(
            get_user_dir('fig'),
            'ae', 'D=%02i_latent-manipulation_%s_%s-crop.png' %
                  (hparams['n_ae_latents'], hparams['session'], crop_type))
    else:
        save_file = None

    if plot_func.__name__ == 'plot_2d_frame_array':
        """plot generated frames and differences separately"""

        # plot generated frames
        if crop_type:
            plot_func(ims, markers=None, marker_kwargs=marker_kwargs, save_file=save_file)
        else:
            plot_func(ims, markers=markers, marker_kwargs=marker_kwargs, save_file=save_file)

        # plot differences
        if n_latents == 2 and interp_func.__name__ == 'interpolate_2d':
            # use top-most row as base frame for each column
            ims_diff = [[None for _ in range(n_frames)] for _ in range(n_frames)]
            for r, ims_list_y in enumerate(ims):
                for c, im in enumerate(ims_list_y):
                    ims_diff[r][c] = 0.5 + (im - ims[0][c])  # compare across cols
            plot_func(
                ims_diff, markers=markers, marker_kwargs=marker_kwargs, save_file=save_file,
                **kwargs)

        # use left-most column as base frame for each row
        ims_diff = [[None for _ in range(n_frames)] for _ in range(len(ims))]
        for r, ims_list_y in enumerate(ims):
            for c, im in enumerate(ims_list_y):
                ims_diff[r][c] = 0.5 + (im - ims[r][0])  # compare across rows
        plot_func(
            ims_diff, markers=markers, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)

    else:
        """plot generated frames and differences together"""
        if crop_type:
            raise NotImplementedError
        else:
            plot_func(
                ims, markers=None, marker_kwargs=marker_kwargs, save_file=save_file, **kwargs)


def get_cluster_prototype_ims(dataset, n_clusters, as_numpy=False):

    import pickle
    from sklearn.cluster import KMeans

    # ----------------------
    # load AE model
    # ----------------------

    if dataset == 'ibl':
        lab = 'ibl'
        expt = 'ephys'
        iters = 200
        frac = '0.5'
        n_ae_latents = 6
    elif dataset == 'dipoppa':
        lab = 'dipoppa'
        expt = 'pupil'
        iters = 200
        frac = '0.5'
        n_ae_latents = 5
    elif dataset == 'musall':
        lab = 'musall'
        expt = 'vistrained'
        iters = 200
        frac = '0.5'
        n_ae_latents = 7
    else:
        raise Exception

    # set model info
    version = 'best'  # test-tube version; 'best' finds the version with the lowest mse
    sess_idx = 0  # when using a multisession, this determines which session is used
    hparams = {
        'data_dir': get_user_dir('data'),
        'save_dir': get_user_dir('save'),
        'experiment_name': 'iters-%i_frac-%s' % (iters, frac),
        'model_class': 'vae',
        'model_type': 'conv',
        'n_ae_latents': n_ae_latents,
        'rng_seed_data': 0,
        'trial_splits': '8;1;1;0',
        'train_frac': float(frac),
        'rng_seed_model': 0,
        'conditional_encoder': False,
    }

    # programmatically fill out other hparams options
    get_lab_example(hparams, lab, expt)
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)

    # build model(s)
    if hparams['model_class'] == 'ae':
        from behavenet.models import AE as Model
    elif hparams['model_class'] == 'vae':
        from behavenet.models import VAE as Model
    else:
        raise NotImplementedError
    model_ae, data_generator = get_best_model_and_data(hparams, Model, version=version)

    # ----------------------
    # cluster latents
    # ----------------------

    # load latents
    sess_id = str('%s_%s_%s_%s_latents.pkl' % (
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))
    filename = os.path.join(
        hparams['expt_dir'], 'version_%i' % 0, sess_id)
    if not os.path.exists(filename):
        print('exporting latents...', end='')
        from behavenet.fitting.eval import export_latents
        export_latents(data_generator, model_ae)
        print('done')
    latent_dict = pickle.load(open(filename, 'rb'))
    # get all test latents
    dtype = 'test'
    latents = []
    trials = []
    frames = []
    for trial in latent_dict['trials'][dtype]:
        ls = latent_dict['latents'][trial]
        n_frames_batch = ls.shape[0]
        latents.append(ls)
        trials.append([trial] * n_frames_batch)
        frames.append(np.arange(n_frames_batch))
        # print('trial: %i, frames: %i' % (trial, n_frames_batch))
    latents = np.concatenate(latents)
    trials = np.concatenate(trials)
    frames = np.concatenate(frames)

    np.random.seed(0)  # to reproduce clusters
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    distances = kmeans.fit_transform(latents)
    clust_id = kmeans.predict(latents)

    # ----------------------
    # get representative example from each cluster
    # ----------------------

    example_idxs = []
    trial_idxs = []
    frame_idxs = []
    ims = []
    # for clust in range(n_clusters):
    for clust in range(n_clusters):

        # get any frame in this cluster
        # frame_idx = np.where(clust_ids==clust)[0][0]

        # get frame that is closest to cluster center
        frame_idx = np.argmin(distances[:, clust])
        example_idxs.append(frame_idx)

        trial_curr = trials[frame_idx]
        frame_curr = frames[frame_idx]

        batch = data_generator.datasets[0][trial_curr]
        if as_numpy:
            im = batch['images'].cpu().detach().numpy()[frame_curr, 0]
        else:
            im = batch['images'][None, frame_curr]

        trial_idxs.append(trial_curr)
        frame_idxs.append(frame_curr)
        ims.append(im)

    return example_idxs, trial_idxs, frame_idxs, ims


def interpolate_points(points, n_frames):
    """Scale arbitrary points"""
    n_points = len(points)
    if isinstance(n_frames, int):
        n_frames = [n_frames] * (n_points - 1)
    assert len(n_frames) == (n_points - 1)
    inputs_list = []
    for p in range(n_points - 1):
        p0 = points[None, p]
        p1 = points[None, p + 1]
        p_vec = (p1 - p0) / n_frames[p]
        for pn in range(n_frames[p]):
            vec = p0 + pn * p_vec
            inputs_list.append(vec)
    return inputs_list


def interpolate_point_path(
        interp_type, model, ims_0, latents_0, labels_0, points, n_frames=10, ch=0,
        crop_kwargs=None, apply_inverse_transform=True):
    """Return reconstructed images created by interpolating through multiple points.

    Parameters
    ----------
    interp_type : :obj:`str`
        'latents' | 'labels'
    model : :obj:`behavenet.models` object
        autoencoder model
    ims_0 : :obj:`np.ndarray`
        base images for interpolating labels, of shape (1, n_channels, y_pix, x_pix)
    latents_0 : :obj:`np.ndarray`
        base latents of shape (1, n_latents); these values will be used if
        `interp_type='labels'`, and they will be ignored if `inter_type='latents'`
        (since `points` will be used)
    labels_0 : :obj:`np.ndarray`
        base labels of shape (1, n_labels); these values will be used if
        `interp_type='latents'`, and they will be ignored if `inter_type='labels'`
        (since `points` will be used)
    points : :obj:`list`
        one entry for each point in path; each entry is an np.ndarray of shape (n_latents,)
    n_frames : :obj:`int` or :obj:`array-like`
        number of interpolation points between each point; can be an integer that is used
        for all paths, or an array/list of length one less than number of points
    ch : :obj:`int`, optional
        specify which channel of input images to return (can only be a single value)

    Returns
    -------
    :obj:`tuple`
        - ims_list (:obj:`list` of :obj:`np.ndarray`) interpolated images
        - inputs_list (:obj:`list` of :obj:`np.ndarray`) interpolated values

    """

    if model.hparams.get('conditional_encoder', False):
        raise NotImplementedError

    n_points = len(points)
    if isinstance(n_frames, int):
        n_frames = [n_frames] * (n_points - 1)
    assert len(n_frames) == (n_points - 1)

    ims_list = []
    inputs_list = []

    for p in range(n_points - 1):

        p0 = points[None, p]
        p1 = points[None, p + 1]
        p_vec = (p1 - p0) / n_frames[p]

        for pn in range(n_frames[p]):

            vec = p0 + pn * p_vec

            if interp_type == 'latents':

                if model.hparams['model_class'] == 'cond-ae' \
                        or model.hparams['model_class'] == 'cond-vae':
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=apply_inverse_transform,
                        labels=torch.from_numpy(labels_0).float().to(model.hparams['device']))
                else:
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=apply_inverse_transform)

            elif interp_type == 'labels':

                if model.hparams['model_class'] == 'cond-ae-msp' \
                        or model.hparams['model_class'] == 'sss-vae':
                    im_tmp = get_reconstruction(
                        model, vec, apply_inverse_transform=True)
                else:  # cond-ae
                    im_tmp = get_reconstruction(
                        model, ims_0,
                        labels=torch.from_numpy(vec).float().to(model.hparams['device']))
            else:
                raise NotImplementedError

            if crop_kwargs is not None:
                if not isinstance(ch, int):
                    raise ValueError('"ch" must be an integer to use crop_kwargs')
                ims_list.append(get_crop(
                    im_tmp[0, ch],
                    crop_kwargs['y_0'], crop_kwargs['y_ext'],
                    crop_kwargs['x_0'], crop_kwargs['x_ext']))
            else:
                if isinstance(ch, int):
                    ims_list.append(np.copy(im_tmp[0, ch]))
                else:
                    ims_list.append(np.copy(concat(im_tmp[0])))

            inputs_list.append(vec)

    return ims_list, inputs_list


def make_interpolated(
        ims, save_file, markers=None, text=None, text_title=None, text_color=[1, 1, 1],
        frame_rate=20, scale=3, markersize=10, markeredgecolor='w', markeredgewidth=1, ax=None):

    n_frames = len(ims)
    y_pix, x_pix = ims[0].shape

    if ax is None:
        fig_width = scale / 2
        fig_height = y_pix / x_pix * scale / 2
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        ax = plt.gca()
        return_ims = False
    else:
        return_ims = True

    ax.set_xticks([])
    ax.set_yticks([])

    default_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    txt_kwargs = {
        'fontsize': 4, 'color': text_color, 'fontname': 'monospace',
        'horizontalalignment': 'left', 'verticalalignment': 'center',
        'transform': ax.transAxes}

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i, im in enumerate(ims):
        im_tmp = []
        im_tmp.append(ax.imshow(im, **default_kwargs))
        # [s.set_visible(False) for s in ax.spines.values()]
        if markers is not None:
            im_tmp.append(ax.plot(
                markers[i, 0], markers[i, 1], '.r', markersize=markersize,
                markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)[0])
        if text is not None:
            im_tmp.append(ax.text(0.02, 0.06, text[i], **txt_kwargs))
        if text_title is not None:
            im_tmp.append(ax.text(0.02, 0.92, text_title[i], **txt_kwargs))
        ims_ani.append(im_tmp)

    if return_ims:
        return ims_ani
    else:
        plt.tight_layout(pad=0)
        ani = animation.ArtistAnimation(fig, ims_ani, blit=True, repeat_delay=1000)
        save_movie(save_file, ani, frame_rate=frame_rate)


def make_interpolated_wdist(
        ims, latents, points, save_file, xlim=None, ylim=None, frame_rate=20):
    n_frames = len(ims)
    y_pix, x_pix = ims[0].shape
    n_channels = 1

    scale_ = 4
    fig_width = scale_ * n_channels
    fig_height = y_pix / x_pix * scale_ / 2
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=300)

    # get rid of ticks on video panel
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    default_kwargs = {'animated': True, 'cmap': 'gray', 'vmin': 0, 'vmax': 1}

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i, im in enumerate(ims):

        frames_curr = []

        im_tmp = axes[1].imshow(im, **default_kwargs)
        frames_curr.append(im_tmp)

        fr_tmp0 = axes[0].scatter(
            latents[:, 0], latents[:, 1], c=[[0, 0, 0]], s=0.5, alpha=0.25,
            linewidths=0)
        #         axes[0].invert_yaxis()
        axes[0].set_xlabel('Left paw marker x\n(normalized)', fontsize=8)
        axes[0].set_ylabel('Left paw marker y\n(normalized)', fontsize=8)
        if xlim is not None:
            axes[0].set_xlim(xlim)
        if ylim is not None:
            axes[0].set_ylim(ylim)
        frames_curr.append(fr_tmp0)

        fr_tmp1 = axes[0].plot(
            points[:, 0], points[:, 1], 'sr', markersize=1, markeredgecolor='r')[0]
        #         axes[0].invert_yaxis()
        frames_curr.append(fr_tmp1)

        fr_tmp2 = axes[0].plot(
            points[i, 0], points[i, 1], 'sr', markersize=3, markeredgecolor='r')[0]
        #         axes[0].invert_yaxis()
        frames_curr.append(fr_tmp2)

        ims_ani.append(frames_curr)

    plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims_ani, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


def make_interpolated_multipanel(
        ims, save_file, markers=None, text=None, text_title=None,
        frame_rate=20, n_cols=3, scale=1, **kwargs):
    n_panels = len(ims)

    markers = [None] * n_panels if markers is None else markers
    text = [None] * n_panels if text is None else text

    y_pix, x_pix = ims[0][0].shape
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_width = scale / 2 * n_cols
    fig_height = y_pix / x_pix * scale / 2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

    # ims is a list of lists, each row is a list of artists to draw in the current frame; here we
    # are just animating one artist, the image, in each frame
    ims_ani = []
    for i, (ims_curr, markers_curr, text_curr) in enumerate(zip(ims, markers, text)):
        col = i % n_cols
        row = int(np.floor(i / n_cols))
        if i == 0:
            text_title_str = text_title
        else:
            text_title_str = None
        ims_ani_curr = make_interpolated(
            ims=ims_curr, markers=markers_curr, text=text_curr, text_title=text_title_str,
            ax=axes[row, col], save_file=None, **kwargs)
        ims_ani.append(ims_ani_curr)

    # turn off other axes
    i += 1
    while i < n_rows * n_cols:
        col = i % n_cols
        row = int(np.floor(i / n_cols))
        axes[row, col].set_axis_off()
        i += 1

    # rearrange ims:
    # currently a list of length n_panels, each element of which is a list of length n_t
    # we need a list of length n_t, each element of which is a list of length n_panels
    n_frames = len(ims_ani[0])
    ims_final = [[] for _ in range(n_frames)]
    for i in range(n_frames):
        for j in range(n_panels):
            ims_final[i] += ims_ani[j][i]

    #     plt.tight_layout(pad=0)

    ani = animation.ArtistAnimation(fig, ims_final, blit=True, repeat_delay=1000)
    save_movie(save_file, ani, frame_rate=frame_rate)


# --------------------------------------
# disentangling functions
# --------------------------------------

def compute_latent_std(hparams, version, sess_id=0, dtype='val'):
    import pickle

    # load latents
    sess_id = str('%s_%s_%s_%s_latents.pkl' % (
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session']))
    filename = os.path.join(
        hparams['expt_dir'], 'version_%i' % version, sess_id)
    if not os.path.exists(filename):
        #         print('exporting latents...', end='')
        #         export_latents(data_generator, model_ae)
        #         print('done')
        raise NotImplementedError
    latent_dict = pickle.load(open(filename, 'rb'))
    print('loaded latents from %s' % filename)
    # get all test latents
    latents = []
    for trial in latent_dict['trials'][dtype]:
        ls = latent_dict['latents'][trial]
        latents.append(ls)
    latents = np.concatenate(latents)

    return np.std(latents, axis=0)


def compute_metric(
        ims, model, n_ae_latents, latents_std, std_range=1, L=20, dist='uniform'):
    # L: number of random samples in latent space

    if model.hparams['model_class'] == 'sss-vae':
        n_labels = model.hparams.get('n_labels', 0)
    else:
        n_labels = 0

    lowest_vars = np.zeros((n_ae_latents, n_ae_latents))

    for im in ims:

        # step 4: push a sample frame $x_k$ through the encoder
        if model.hparams['model_class'] == 'sss-vae':
            y, w, logvar, pi, outsize = model.encoding(im.to(model.hparams['device']))
            y_np = y.cpu().detach().numpy()
            w_np = w.cpu().detach().numpy()
            logvar_np = logvar.cpu().detach().numpy()
        elif model.hparams['model_class'] == 'beta-tcvae':
            mu, logvar, pi, outsize = model.encoding(im.to(model.hparams['device']))
            mu_np = mu.cpu().detach().numpy()
            logvar_np = logvar.cpu().detach().numpy()
        elif model.hparams['model_class'] == 'cond-vae':
            # z_hat_, _, _, _ = model.encoding(im.to(model.hparams['device']))
            # z_hat = z_hat_.cpu().detach().numpy()
            raise NotImplementedError
        else:
            raise NotImplementedError

        for d in range(n_ae_latents):

            w_hats = np.zeros((L, n_ae_latents))

            for l in range(L):

                # step 5: fix latent dim $d$ and sample the remaining dims from a
                # uniform/normal/posterior distribution
                idxs_ = np.arange(n_ae_latents + n_labels)
                idxs = idxs_[idxs_ != (n_labels + d)]

                # sample all but dim $d$ (super+unsupervised dims)
                if model.hparams['model_class'] == 'sss-vae':
                    # sample all unsupervised dims but dim $d$
                    # n_idxs = np.concatenate([np.arange(n_labels), [n_labels + d]])
                    # idxs = np.where(~np.in1d(idxs_, n_idxs))[0]
                    z_hat_np = np.concatenate([y_np, w_np], axis=1).astype('float32')
                else:
                    z_hat_np = np.copy(mu_np)

                # uniform distribution
                if dist == 'uniform':
                    # sample latents from range [-std_range*s, +std_range*s]
                    latents_range = 2 * std_range * latents_std
                    latents_offset = -std_range * latents_std
                    eps = np.random.random((1, n_ae_latents + n_labels))
                    sample = latents_range * eps + latents_offset
                elif dist == 'normal':
                    eps = np.random.randn(1, n_ae_latents + n_labels)
                    sample = latents_std * np.sqrt(std_range) * eps
                elif dist == 'posterior':
                    eps = np.random.randn(1, n_ae_latents + n_labels)
                    sample = z_hat_np + np.sqrt(np.exp(logvar_np)) * eps
                else:
                    raise NotImplementedError

                # update latents with sampled values
                z_hat_np[0, idxs] = sample[0, idxs]

                # step 6: push this new latent vector through the decoder and back
                # through the encoder to get the updated latent vector
                z_hat = from_numpy(z_hat_np).to(model.hparams['device'])
                im_hat = model.decoding(z_hat, pi, outsize)
                if model.hparams['model_class'] == 'sss-vae':
                    _, w_hat, _, _, _ = model.encoding(im_hat)
                elif model.hparams['model_class'] == 'beta-tcvae':
                    w_hat, _, _, _ = model.encoding(im_hat)
                else:
                    raise NotImplementedError
                w_hats[l, :] = w_hat.cpu().detach().numpy()

            # step 8: divide the $L$ latent representations by their standard deviation $s$
            w_hats /= latents_std[n_labels:]

            # step 9: record the dimension with the smallest variance across the $L$ samples
            idx_min_var = np.argmin(np.var(w_hats, axis=0))
            lowest_vars[d, idx_min_var] += 1
    #             lowest_vars[d] += np.var(w_hats, axis=0)

    error_rate = 1 - np.sum(np.diag(lowest_vars)) / (len(ims) * n_ae_latents)
    return lowest_vars, error_rate


def compute_metric_scan(
        n_scans, model, n_ae_latents, latents_std, std_range=1, L=20, dist='uniform'):
    # L: number of random samples in latent space

    n_labels = model.hparams.get('n_labels', 0)

    ranges = [np.linspace(-std_range * s, std_range * s, n_scans) for s in latents_std]
    lowest_vars = np.zeros((n_ae_latents, n_ae_latents))

    for s in range(n_scans):
        for d in range(n_ae_latents):
            w_hats = np.zeros((L, n_ae_latents))
            for l in range(L):
                if dist == 'uniform':
                    # sample latents from range [-std_range*s, +std_range*s]
                    latents_range = 2 * std_range * latents_std
                    latents_offset = -std_range * latents_std
                    eps = np.random.random((1, n_ae_latents + n_labels))
                    sample = latents_range * eps + latents_offset
                elif dist == 'normal':
                    eps = np.random.randn(1, n_ae_latents + n_labels)
                    sample = latents_std * np.sqrt(std_range) * eps
                else:
                    raise NotImplementedError

                # update latents with sampled values
                z_hat_np = sample
                z_hat_np[0, d] = ranges[d][s]

                # step 6: push this new latent vector through the decoder and back
                # through the encoder to get the updated latent vector
                z_hat = from_numpy(z_hat_np.astype('float32')).to(model.hparams['device'])
                im_hat = model.decoding(z_hat, None, None)
                _, w_hat, _, _, _ = model.encoding(im_hat)
                w_hats[l, :] = w_hat.cpu().detach().numpy()

            # step 8: divide the $L$ latent representations by their standard deviation $s$
            w_hats /= latents_std[n_labels:]

            # step 9: record the dimension with the smallest variance across the $L$ samples
            idx_min_var = np.argmin(np.var(w_hats, axis=0))
            lowest_vars[d, idx_min_var] += 1
    #             lowest_vars[d] += np.var(w_hats, axis=0)

    error_rate = 1 - np.sum(np.diag(lowest_vars)) / (n_scans * n_ae_latents)
    return lowest_vars, error_rate


def compute_metric_traversal(
        ims, model, n_ae_latents, latents_std, latent_range, label_range=None, L=20):
    # L: number of random samples in latent space

    if model.hparams['model_class'] == 'sss-vae':
        n_labels = model.hparams.get('n_labels', 0)
    else:
        n_labels = 0

    mses = np.zeros((n_ae_latents, n_ae_latents, len(ims)))

    for i, im in enumerate(ims):

        # step 4: push a sample frame $x_k$ through the encoder
        if model.hparams['model_class'] == 'sss-vae':
            y, w, logvar, pi, outsize = model.encoding(im.to(model.hparams['device']))
            y_np = y.cpu().detach().numpy()
            w_np = w.cpu().detach().numpy()
            logvar_np = logvar.cpu().detach().numpy()
        elif model.hparams['model_class'] == 'beta-tcvae':
            mu, logvar, pi, outsize = model.encoding(im.to(model.hparams['device']))
            mu_np = mu.cpu().detach().numpy()
            logvar_np = logvar.cpu().detach().numpy()
        elif model.hparams['model_class'] == 'cond-vae':
            # z_hat_, _, _, _ = model.encoding(im.to(model.hparams['device']))
            # z_hat = z_hat_.cpu().detach().numpy()
            raise NotImplementedError
        else:
            raise NotImplementedError

        # compute change in reconstructed z when manipulating single dim of original z
        for d in range(n_ae_latents):

            if model.hparams['model_class'] == 'sss-vae':
                z_hat_np = np.concatenate([y_np, w_np], axis=1).astype('float32')
            else:
                z_hat_np = np.copy(mu_np)

            points = np.array([z_hat_np] * 2)

            if d < n_labels:
                points[0, d] = label_range['min'][d]
                points[1, d] = label_range['max'][d]
                ims_re, inputs = interpolate_point_path(
                    'labels', model, im.to(model.hparams['device']), None, None, points=points,
                    n_frames=L)
            else:
                points[0, d] = latent_range['min'][d]
                points[1, d] = latent_range['max'][d]
                ims_re, inputs = interpolate_point_path(
                    'latents', model, im.to(model.hparams['device']), None, None, points=points,
                    n_frames=L)

            zs_og = np.vstack(inputs)

            inputs_re = get_latents(ims_re, model)
            zs_re = np.vstack(inputs_re)

            mses[d, :, i] = np.mean(np.square(zs_og - zs_re)) / (latents_std ** 2)

    return np.mean(mses, axis=2)


def get_latents(ims, model):
    use_mean = True
    dataset = 0
    zs = []
    for im in ims:
        #         _, latents = get_reconstruction(model, im[None, None, ...], return_latents=True)
        if not isinstance(im, torch.Tensor):
            im = torch.Tensor(im[None, None, ...])
        if model.hparams['model_class'] == 'cond-ae-msp':
            ims_recon, latents, _ = model(im, dataset=dataset)
        elif model.hparams['model_class'] == 'vae' \
                or model.hparams['model_class'] == 'beta-tcvae':
            ims_recon, latents, _, _ = model(im, dataset=dataset, use_mean=use_mean)
        elif model.hparams['model_class'] == 'sss-vae':
            ims_recon, _, latents, _, yhat = model(im, dataset=dataset, use_mean=use_mean)
        elif model.hparams['model_class'] == 'cond-ae':
            ims_recon, latents = model(im, dataset=dataset, labels=labels, labels_2d=labels_2d)
        elif model.hparams['model_class'] == 'cond-vae':
            ims_recon, latents, _, _ = model(im, dataset=dataset, labels=labels,
                                             labels_2d=labels_2d)
        else:
            raise ValueError('Invalid model class %s' % model.hparams['model_class'])
        latents = latents.cpu().detach().numpy()
        if model.hparams['model_class'] == 'sss-vae':
            yhat = yhat.cpu().detach().numpy()
            latents[:, :model.hparams['n_labels']] = yhat

        zs.append(latents)
    return zs

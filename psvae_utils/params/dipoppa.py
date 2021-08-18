"""PS-VAE analysis params for Dipoppa data."""

params_dict = {

    'lab': 'dipoppa',
    'expt': 'pupil',
    'animal': 'MD0ST5',
    'session': 'session-3',
    'n_labels': 3,
    'label_names': ['Pupil area', 'Pupil (y)', 'Pupil (x)'],

    # best model
    'best_alpha': 1000,
    'best_beta': 20,
    'best_gamma': 1000,
    'best_rng': 0,

    # hyper search
    'n_ae_latents': [2, 4, 8, 16],
    'alpha_weights': [100, 500, 1000, 5000, 10000],
    'alpha_train_frac': '0.5',
    'beta_weights': [1, 5, 10, 20],
    'gamma_weights': [0, 100, 500, 1000],
    'beta_gamma_train_frac': '0.5',
    'alpha': 1000,

    # label reconstructions
    'label_recon_trials': [43, 83, 73],  # [23, 133, 143],
    'xtick_locs': [0, 30, 60, 90, 120, 150],
    'frame_rate': 30,
    'scale': 0.45,

    # latent traversals
    'label_min_p': 5,
    'label_max_p': 95,
    'ch': 0,
    'n_frames_zs': 4,
    'n_frames_zu': 4,
    'label_idxs': [1, 2],  # pupil location
    'crop_type': 'fixed',  # crop around pupil for supervised latents
    'crop_kwargs': {'y_0': 48, 'y_ext': 48, 'x_0': 192, 'x_ext': 64},
    'trial_idxs': [11, None, 21],
    'trials': [None, 393, None],
    'batch_idxs': [60, 27, 99],
    'n_cols': 3,
    'text_color': [0, 0, 0],
}

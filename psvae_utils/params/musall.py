"""PS-VAE analysis params for Musall data (no paw tracking)."""

params_dict = {

    'lab': 'musall',
    'expt': 'vistrained',
    'animal': 'mSM36',
    'session': '05-Dec-2017',
    'n_labels': 3,
    'label_names': ['Levers', 'L Spout', 'R Spout'],

    # best model
    'best_alpha': 1000,
    'best_beta': 1,
    'best_gamma': 1000,
    'best_rng': 2,

    # hyper search
    'n_ae_latents': [2, 4, 6],
    'alpha_weights': [50, 100, 500, 1000, 5000, 10000],
    'alpha_train_frac': '0.5',
    'beta_weights': [1, 5, 10, 20],
    'gamma_weights': [0, 100, 500, 1000],
    'beta_gamma_train_frac': '0.5',
    'alpha': 1000,

    # label reconstructions
    'label_recon_trials': [9, 19, 29, 189],  # [59, 189],
    'xtick_locs': [0, 60, 120, 180],
    'frame_rate': 30,
    'scale': 0.25,

    # latent traversals
    'label_min_p': 1,
    'label_max_p': 99,
    'ch': 1,
    'n_frames_zs': 2,
    'n_frames_zu': 4,
    'label_idxs': [1, 2],
    'crop_type': None,
    'crop_kwargs': None,
    'trial_idxs': [11, 11, 11, 5],
    'trials': [None, None, None, None],
    'batch_idxs': [99, 0, 50, 180],
    'n_cols': 2,
    'text_color': [1, 1, 1],
}

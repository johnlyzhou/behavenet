"""PS-VAE analysis params for IBL data."""

params_dict = {

    'lab': 'ibl',
    'expt': 'ephys',
    'animal': 'animal-0',
    'session': '192_coarse-66',
    'n_labels': 4,
    'label_names': ['L paw (x)', 'R paw (x)', 'L paw (y)', 'R paw (y)'],

    # best model
    'best_alpha': 1000,
    'best_beta': 5,
    'best_gamma': 500,
    'best_rng': 0,

    # hyper search
    'n_ae_latents': [2, 4, 8, 16],
    'alpha_weights': [50, 100, 500, 1000, 5000, 10000],
    'alpha_train_frac': '1.0',
    'beta_weights': [1, 5, 10, 20],
    'gamma_weights': [0, 100, 500, 1000],
    'beta_gamma_train_frac': '0.5',
    'alpha': 1000,

    # label reconstructions
    'label_recon_trials': [229, 289, 419],
    'xtick_locs': [0, 30, 60, 90],
    'frame_rate': 60,
    'scale': 0.4,

    # latent traversals
    'label_min_p': 35,
    'label_max_p': 85,
    'ch': 0,
    'n_frames_zs': 4,
    'n_frames_zu': 4,
    'label_idxs': [1, 0],  # horizontally move left/right paws
    'crop_type': None,
    'crop_kwargs': None,
    'trial_idxs': [11, 4, 0, None, None, None, None],  # None
    'trials': [None, None, None, 169, 129, 429, 339],  # 649
    'batch_idxs': [99, 99, 99, 16, 46, 11, 79],  # 61
    'n_cols': 3,
    'text_color': [1, 1, 1],
}

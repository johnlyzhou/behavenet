"""PS-VAE analysis params for IBL 2D data (with paw tracking)."""

params_dict = {

    'lab': 'single-view',
    'expt': 'ephys',
    'animal': None,
    'session': None,
    'raw_data_path': '/media/mattw/ibl/raw_data/',
    'n_labels': 4,
    'label_names': ['L paw (x)', 'R paw (x)', 'L paw (y)', 'R paw (y)'],

    # best model
    'best_alpha': 50,
    'best_beta': 10,
    'best_gamma': None,
    'best_delta': 50,
    'best_rng': 3,

    # hyper search
    'n_ae_latents': [2, 4, 8, 16],
    'alpha_weights': [10, 50, 500, 1000],
    'alpha_train_frac': '1.0',
    'beta_weights': [1, 5, 10, 20],
    'delta_weights': [10, 50, 100, 500],
    'beta_delta_train_frac': '1.0',
    'alpha': 50,  # what is this for?
    'batch_size': 96,

    # label reconstructions
    'label_recon_trials': [19, 29, 39],
    'xtick_locs': [0, 30, 60, 90],
    'frame_rate': 60,
    'scale': 0.15,

    # latent traversals
    'label_min_p': 10,
    'label_max_p': 90,
    'ch': 0,
    'n_frames_zs': 4,
    'n_frames_zu': 4,
    'label_idxs': [0, 1, 2, 3],
    'crop_type': None,
    'crop_kwargs': None,
    'trial_idxs': None,
    'trials': None,
    'batch_idxs': None,
    'n_cols': 3,
    'text_color': [1, 1, 1],
}
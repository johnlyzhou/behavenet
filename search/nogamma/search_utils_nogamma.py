import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import r2_score
from behavenet import get_user_dir
from behavenet.fitting.eval import export_latents
from behavenet.fitting.utils import (
    get_expt_dir,
    get_session_dir,
    get_lab_example
)


def apply_masks(data, masks):
    return data[masks == 1]


def get_expt_dir_wrapper(lab, expt, animal, session, expt_name, n_ae_latents):
    hparams = get_psvae_hparams()
    get_lab_example(hparams, lab, expt)
    hparams['experiment_name'] = expt_name
    hparams['n_ae_latents'] = n_ae_latents
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    return get_expt_dir(hparams)


def get_version_dir(lab, expt, animal, session, expt_name, n_ae_latents, alpha, beta):
    hparams = get_psvae_hparams()
    get_lab_example(hparams, lab, expt)
    hparams['experiment_name'] = expt_name
    hparams['n_ae_latents'] = n_ae_latents
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    for version_dir in os.listdir(hparams['expt_dir']):
        filename = os.path.join(hparams['expt_dir'], version_dir, 'meta_tags.pkl')
        if os.path.exists(filename):
            meta_tags = pkl.load(open(filename, 'rb'))
            if alpha == meta_tags['ps_vae.alpha'] and beta == meta_tags['ps_vae.beta']:
                return os.path.join(hparams['expt_dir'], version_dir)
    print("Version does not exist for alpha: {} and beta: {}".format(alpha, beta))


def get_psvae_hparams(**kwargs):
    hparams = {
        'data_dir': get_user_dir('data'),
        'save_dir': get_user_dir('save'),
        'model_class': 'ps-vae',
        'model_type': 'conv',
        'rng_seed_data': 0,
        'trial_splits': '8;1;1;0',
        'train_frac': 1,
        'rng_seed_model': 0,
        'fit_sess_io_layers': False,
        'learning_rate': 1e-4,
        'l2_reg': 0,
        'conditional_encoder': False,
        'vae.beta': 1}
    # update based on kwargs
    for key, val in kwargs.items():
        if key == 'alpha' or key == 'beta':
            hparams['ps_vae.%s' % key] = val
        else:
            hparams[key] = val
    return hparams


def get_meta_tags(expt_dir, version):
    filename = os.path.join(expt_dir, 'version_{}'.format(version), 'meta_tags.pkl')
    try:
        meta_tags = pkl.load(open(filename, 'rb'))
        return meta_tags
    except OSError as e:
        print(e)


def list_hparams(lab, expt, animal, session, expt_name, n_ae_latents):
    hparams = get_psvae_hparams()
    get_lab_example(hparams, lab, expt)
    hparams['experiment_name'] = expt_name
    hparams['n_ae_latents'] = n_ae_latents
    hparams['animal'] = animal
    hparams['session'] = session
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['expt_dir'] = get_expt_dir(hparams)
    alphas = set()
    betas = set()
    for version_dir in os.listdir(hparams['expt_dir']):
        if 'version' in version_dir:
            filename = os.path.join(hparams['expt_dir'], version_dir, 'meta_tags.pkl')
            if os.path.exists(filename):
                meta_tags = pkl.load(open(filename, 'rb'))
                alphas.add(meta_tags['ps_vae.alpha'])
                betas.add(meta_tags['ps_vae.beta'])
    return sorted(list(alphas)), sorted(list(betas))


def get_label_r2(hparams, model, data_generator, version, label_names, dtype='val', overwrite=False):
    save_file = os.path.join(
        hparams['expt_dir'], 'version_%i' % version, 'r2_supervised.csv'
    )
    if not os.path.exists(save_file) or overwrite:
        if not os.path.exists(save_file):
            print('R^2 metrics do not exist; computing from scratch')
        else:
            print('Overwriting metrics at %s' % save_file)
        metrics_df = []
        data_generator.reset_iterators(dtype)
        for _ in tqdm(range(data_generator.n_tot_batches[dtype])):
            # get next minibatch and put it on the device
            data, sess = data_generator.next_batch(dtype)
            x = data['images'][0]
            y = data['labels'][0].cpu().detach().numpy()
            if 'labels_masks' in data:
                n = data['labels_masks'][0].cpu().detach().numpy()
            else:
                n = np.ones_like(y)
            z = model.get_transformed_latents(x, dataset=sess)
            for i in range(len(label_names)):
                y_true = apply_masks(y[:, i], n[:, i])
                y_pred = apply_masks(z[:, i], n[:, i])
                if len(y_true) > 10:
                    r2 = r2_score(y_true, y_pred,
                                  multioutput='variance_weighted')
                    mse = np.mean(np.square(y_true - y_pred))
                else:
                    r2 = np.nan
                    mse = np.nan
                metrics_df.append(pd.DataFrame({
                    'Trial': data['batch_idx'].item(),
                    'Label': label_names[i],
                    'R2': r2,
                    'MSE': mse,
                    'Model': 'PS-VAE'}, index=[0]))

        metrics_df = pd.concat(metrics_df)
        print('Saving results to %s' % save_file)
        metrics_df.to_csv(save_file, index=False, header=True)
    else:
        print('Loading results from %s' % save_file)
        metrics_df = pd.read_csv(save_file)
    return metrics_df


def load_latents_trials_frames(hparams, data_generator, model_ae=None, dtype='test'):
    sess_id = '{}_{}_{}_{}_latents.pkl'.format(
        hparams['lab'], hparams['expt'], hparams['animal'], hparams['session'])
    filename = os.path.join(
        hparams['expt_dir'], 'version_{}'.format(0), sess_id)
    if not os.path.exists(filename):
        print('Exporting latents...', end='')
        export_latents(data_generator, model_ae)
        print('Done')
    latent_dict = pkl.load(open(filename, 'rb'))
    # get all test latents
    latents = []
    trials = []
    frames = []
    for trial in latent_dict['trials'][dtype]:
        ls = latent_dict['latents'][trial]
        n_frames_batch = ls.shape[0]
        latents.append(ls)
        trials.append([trial] * n_frames_batch)
        frames.append(np.arange(n_frames_batch))
    latents = np.concatenate(latents)
    trials = np.concatenate(trials)
    frames = np.concatenate(frames)
    return latents, trials, frames

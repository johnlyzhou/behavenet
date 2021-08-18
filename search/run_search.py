import copy
import argparse

import commentjson

from search_utils import get_psvae_hparams
from behavenet.fitting.utils import (
    get_session_dir,
    get_lab_example
)
from behavenet.data.utils import build_data_generator
from plot_search import plot_and_film_best


def get_paths(home_dir, config_path):
    with open(config_path) as file:
        config = commentjson.load(file)
    arch_path = home_dir + "/.behavenet/" + config['architecture']
    compute_path = home_dir + "/.behavenet/" + config['compute']
    model_path = home_dir + "/.behavenet/" + config['model']
    training_path = home_dir + "/.behavenet/" + config['training']
    params_path = home_dir + "/.behavenet/" + config['params']

    return arch_path, compute_path, model_path, training_path, params_path


def generate_search_args(home_dir, config_path):
    arch_path, compute_path, model_path, training_path, params_path = get_paths(home_dir, config_path)

    with open(params_path) as file:
        params = commentjson.load(file)
    with open(compute_path) as file:
        compute = commentjson.load(file)
    with open(model_path) as file:
        model = commentjson.load(file)
    with open(config_path) as file:
        config = commentjson.load(file)
    with open(training_path) as file:
        training = commentjson.load(file)

    hparams = get_psvae_hparams()
    get_lab_example(hparams, params['lab'], params['expt'])
    hparams['experiment_name'] = model['experiment_name']
    hparams['n_ae_latents'] = model['n_ae_latents']
    hparams['animal'] = params['animal']
    hparams['session'] = params['session']
    hparams['session_dir'], sess_ids = get_session_dir(hparams)
    hparams['rng_seed_model'] = model['rng_seed_model']
    hparams['model_class'] = model['model_class']
    hparams['trials'] = [0, 1, 2]
    hparams['n_clusters'] = 5
    hparams['device'] = compute['device']
    hparams['as_numpy'] = training['as_numpy']
    hparams['batch_load'] = training['batch_load']
    hparams['label_names'] = generate_labels(hparams, sess_ids)
    hparams['n_labels'] = len(hparams['label_names'])
    hparams['ps_vae.beta'] = model['ps_vae.beta']
    hparams['train_frac'] = training['train_frac']
    return hparams, sess_ids


def generate_labels(hparams, sess_ids):
    if hparams['model_class'] != 'ps-vae':
        raise NotImplementedError('hyperparameter search is only implemented for PS-VAE at this time')

    hparams_new = copy.deepcopy(hparams)

    data_generator = build_data_generator(hparams_new, sess_ids, export_csv=False)

    data, _ = data_generator.next_batch('train')
    n_labels = data['labels'].shape[2]
    generic_label_names = ["label_{}".format(num) for num in range(n_labels)]
    return generic_label_names


def main(args):
    config_path = args.config[0]
    home_dir = args.home_dir[0]
    hparams, sess_ids = generate_search_args(home_dir, config_path)
    lab = hparams['lab']
    expt = hparams['expt']
    animal = hparams['animal']
    session = hparams['session']
    label_names = hparams['label_names']
    experiment_name = hparams['experiment_name']
    n_ae_latents = hparams['n_ae_latents']
    trials = hparams['trials']
    beta_start = hparams['ps_vae.beta'][0]
    model_class = hparams['model_class']
    n_clusters = hparams['n_clusters']
    rng_seed_model = hparams['rng_seed_model']
    rng_seeds_model = None
    save_file = None
    train_frac = hparams['train_frac']
    plot_and_film_best(lab, expt, animal, session, label_names, experiment_name, n_ae_latents, trials,
                       beta_start=beta_start, model_class=model_class, n_clusters=n_clusters,
                       rng_seed_model=rng_seed_model, rng_seeds_model=rng_seeds_model, save_file=save_file,
                       train_frac=train_frac)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", help="path to config.json", nargs=1)
    parser.add_argument(
        "home_dir", help="home directory path", nargs=1)
    arguments = parser.parse_args()

    main(arguments)


# This module uses sample files to create individual PS-VAE model, config, and submit files to run hyperparameter
# searches in parallel across multiple EC2 instances.

import json
import commentjson
import calendar
import time
import os


def create_filename(example_path, out_folder, alpha, beta, gamma):
    example_file = os.path.basename(example_path)
    out_file = 'a{}_b{}_g{}_{}'.format(alpha, beta, gamma, example_file)
    out_path = os.path.join(out_folder, out_file)

    return out_file, out_path


def create_config_json(example_path, out_folder, model_path, alpha, beta, gamma):
    out_file, out_path = create_filename(example_path, out_folder, alpha, beta, gamma)
    print(out_path)

    with open(example_path) as f:
        template = json.load(f)
    template['model'] = model_path

    with open(out_path, 'w') as f:
        json.dump(template, f)

    return out_file


def create_model_json(example_path, out_folder, alpha, beta, gamma):
    out_file, out_path = create_filename(example_path, out_folder, alpha, beta, gamma)
    print(out_path)

    with open(example_path) as f:
        template = commentjson.load(f)

    template['ps_vae.alpha'] = [alpha]
    template['ps_vae.beta'] = [beta]
    template['ps_vae.gamma'] = [gamma]

    with open(out_path, 'w') as f:
        json.dump(template, f)

    return out_file


def create_submit_json(example_path, out_folder, config_path, alpha, beta, gamma):
    out_file, out_path = create_filename(example_path, out_folder, alpha, beta, gamma)
    print(out_path)

    with open(example_path) as f:
        template = commentjson.load(f)

    path = os.path.dirname(template['configname'])
    template['configname'] = os.path.join(path, config_path)
    template['timestamp'] = calendar.timegm(time.gmtime())

    with open(out_path, 'w') as f:
        json.dump(template, f)


def main():
    config_folder = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/config'
    model_folder = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/model'
    submit_folder = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/submit'

    config_path = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/config.json'
    model_path = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/ae_model_rodriguez.json'
    submit_path = '/Users/johnzhou/code/behavenet-ncap-data/rodriguez/submit.json'

    alpha_weights = [100]
    beta_weights = [1, 5, 10, 20]
    gamma_weights = [0, 100, 500, 1000]

    for a in alpha_weights:
        for b in beta_weights:
            for g in gamma_weights:
                model_file = create_model_json(model_path, model_folder, a, b, g)
                config_file = create_config_json(config_path, config_folder, model_file, a, b, g)
                create_submit_json(submit_path, submit_folder, config_file, a, b, g)

                time.sleep(1)


if __name__ == "__main__":
    main()

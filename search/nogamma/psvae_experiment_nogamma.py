from behavenet.fitting.utils import (
    experiment_exists,
    get_expt_dir,
    get_lab_example,
    get_session_dir
)
from nogamma.search_utils_nogamma import (
    get_psvae_hparams,
    get_meta_tags
)


class PSvaeExperiment:
    def __init__(self, lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                 model_class='ps-vae', model_type='conv', **kwargs):
        self.lab = lab
        self.expt = expt
        self.animal = animal
        self.session = session
        self.label_names = label_names
        self.n_labels = len(label_names)
        self.expt_name = expt_name
        self.n_ae_latents = n_ae_latents
        self.alpha = alpha
        self.beta = beta
        self.model_class = model_class
        self.model_type = model_type
        self.hparams, self.version = self.generate_hparams(**kwargs)
        if self.version is None:
            print("Could not find model for alpha=%i, beta=%i" % (
                    alpha, beta))
            raise TypeError
        else:
            self.meta_tags = get_meta_tags(self.hparams['expt_dir'], self.version)

    def generate_hparams(self, **kwargs):
        hparams = get_psvae_hparams()
        hparams['experiment_name'] = self.expt_name
        for key, val in kwargs.items():
            hparams[key] = val
        hparams['n_ae_latents'] = self.n_ae_latents
        get_lab_example(hparams, self.lab, self.expt)
        hparams['animal'] = self.animal
        hparams['session'] = self.session
        hparams['session_dir'], sess_ids = get_session_dir(hparams)
        hparams['expt_dir'] = get_expt_dir(hparams)
        hparams['ps_vae.alpha'] = self.alpha
        hparams['ps_vae.beta'] = self.beta
        _, version = experiment_exists(hparams, which_version=True)
        return hparams, version

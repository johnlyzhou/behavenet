import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from ssm import HMM
from ssm.messages import forward_pass
from scipy.special import logsumexp
from sklearn.metrics import r2_score


# -------------------------------------------------------------------------------------------------
# model fitting functions
# -------------------------------------------------------------------------------------------------

def collect_model_kwargs(
        n_lags_standard, n_lags_sticky, n_lags_recurrent, kappas, observations,
        observation_kwargs={}, hierarchical=False, fit_hmm=False):
    """Collect model kwargs.

    Args:
        n_lags_standard (array-like): number of ar lags for standard transitions
        n_lags_sticky (array-like): number of ar lags for sticky transitions
        n_lags_recurrent (array-like): number of ar lags for recurrent transitions
        kappas (array-like): hyperparam for upweighting diagonal when using sticky transitions
        observations (str): 'ar' | 'diagonal_ar' | 'robust_ar' | 'diagonal_robust_ar'
        observation_kwargs (dict): additional kwargs for obs (e.g. tags for hierarchical models)
        hierarchical (bool): True to fit model with hierarchical observations
        fit_hmm (bool): True to include hmm in collected models

    Returns:
        dict

    """

    model_kwargs = {}

    if hierarchical:
        if len(n_lags_recurrent) > 0 or len(n_lags_sticky) > 0:
            raise NotImplementedError('Cannot fit hierarchical models on recurrent or sticky obs')
        hier_str = 'hierarchical_'
    else:
        hier_str = ''

    # add hmms with standard transitions
    if fit_hmm:
        model_kwargs['hmm'] = {
            'transitions': 'standard',
            'observations': hier_str + 'gaussian',
            'observation_kwargs': observation_kwargs}

    # add models with standard transitions
    for lags in n_lags_standard:
        model_kwargs['arhmm-%i' % lags] = {
            'transitions': 'standard',
            'observations': hier_str + observations,
            'observation_kwargs': {**{'lags': lags}, **observation_kwargs}}

    # add models with sticky transitions
    for lags in n_lags_sticky:
        for kappa in kappas:
            kap = int(np.log10(kappa))
            model_kwargs['arhmm-s%i-%i' % (kap, lags)] = {
                'transitions': 'sticky',
                'transition_kwargs': {'kappa': kappa},
                'observations': hier_str + observations,
                'observation_kwargs': {**{'lags': lags}, **observation_kwargs}}

    # add models with recurrent transitions
    for lags in n_lags_recurrent:
        model_kwargs['rarhmm-%i' % lags] = {
            'transitions': 'recurrent',
            'observations': hier_str + observations,
            'observation_kwargs': {**{'lags': lags}, **observation_kwargs}}

    return model_kwargs


def fit_with_random_restarts(
        K, D, obs, lags, datas, transitions='stationary', tags=None, num_restarts=5, num_iters=100,
        method='em', tolerance=1e-4, save_path=None, init_type='kmeans', dist_mat=None,
        cond_var_A=1e-3, cond_var_V=1e-3, cond_var_b=1e-1, **kwargs):
    all_models = []
    all_lps = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Fit the model with a few random restarts
    for r in range(num_restarts):
        print("Restart ", r)
        np.random.seed(r)
        # build model file
        model_kwargs = {
            'transitions': transitions,
            'observations': obs,
            'observation_kwargs': {'lags': lags},
        }
        model_name = get_model_name(K, model_kwargs)
        save_file = os.path.join(save_path, model_name + '_init-%i.pkl' % r)
        print(save_file)
        if os.path.exists(save_file):
            print('loading results from %s' % save_file)
            with open(save_file, 'rb') as f:
                results = pickle.load(f)
            model = results['model']
            lps = results['lps']
        else:
            observation_kwargs = dict(lags=lags)
            if obs.find('hierarchical') > -1:
                observation_kwargs['cond_variance_A'] = cond_var_A
                observation_kwargs['cond_variance_V'] = cond_var_V
                observation_kwargs['cond_variance_b'] = cond_var_b
                observation_kwargs['cond_dof_Sigma'] = 10
                observation_kwargs['tags'] = np.unique(tags)
            if transitions.find('hierarchical') > -1:
                transition_kwargs = {'tags': np.unique(tags)}
            else:
                transition_kwargs = None
            model = HMM(
                K, D,
                observations=obs, observation_kwargs=observation_kwargs,
                transitions=transitions, transition_kwargs=transition_kwargs)
            init_model(init_type, model, datas, dist_mat=dist_mat)
            lps = model.fit(
                datas, tags=tags, method=method, tolerance=tolerance,
                num_iters=num_iters,  # em
                # num_epochs=num_iters,  # stochastic em
                initialize=False,
                **kwargs)
            results = {'model': model, 'lps': lps}
            with open(save_file, 'wb') as f:
                pickle.dump(results, f)
        all_models.append(model)
        all_lps.append(lps)
    if isinstance(lps, tuple):
        best_model_idx = np.argmax([lps[0][-1] for lps in all_lps])
    else:
        best_model_idx = np.argmax([lps[-1] for lps in all_lps])
    best_model = all_models[best_model_idx]
    best_lps = all_lps[best_model_idx]
    return best_model, best_lps, all_models, all_lps


def init_model(init_type, model, datas, inputs=None, masks=None, tags=None, dist_mat=None):
    """Initialize ARHMM model according to one of several schemes.

    The different schemes correspond to different ways of assigning discrete states to the data
    points; once these states have been assigned, linear regression is used to estimate the model
    parameters (dynamics matrices, biases, covariance matrices)

    * init_type = random: states are randomly and uniformly assigned
    * init_type = kmeans: perform kmeans clustering on data; note that this is not a great scheme
        for arhmms on the fly data, because the fly is often standing still in many different
        poses. These poses will be assigned to different clusters, thus breaking the "still" state
        into many initial states
    * init_type = diff-clust: perform kmeans clustering on differenced data
    * init_type = pca_me: first compute the motion energy of the data (square of differences of
        consecutive time points) and then perform PCA. A threshold applied to the first dimension
        does a reasonable job of separating the data into "moving" and "still" timepoints. All
        "still" timepoints are assigned one state, and the remaining timepoints are clustered using
        kmeans with (K-1) clusters
    * init_type = arhmm: refinement of pca_me approach: perform pca on the data and take top 4
        components (to speed up computation) and fit a 2-state arhmm to roughly split the data into
        "still" and "moving" states (this is itself initialized with pca_me). Then as before the
        moving state is clustered into K-1 states using kmeans.

    Args:
        init_type (str):
            'random' | 'kmeans' | 'pca_me' | 'arhmm'
        model (ssm.HMM object):
        datas (list of np.ndarrays):
        inputs (list of np.ndarrays):
        masks (list of np.ndarrays):
        tags (list of np.ndarrays):

    """

    from ssm.util import one_hot
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from scipy.signal import savgol_filter
    from scipy.stats import norm

    Ts = [data.shape[0] for data in datas]
    K = model.K
    D = model.observations.D
    M = model.observations.M
    lags = model.observations.lags

    if inputs is None:
        inputs = [np.zeros((data.shape[0],) + (M,)) for data in datas]
    elif not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if masks is None:
        masks = [np.ones_like(data, dtype=bool) for data in datas]
    elif not isinstance(masks, (list, tuple)):
        masks = [masks]

    if tags is None:
        tags = [None] * len(datas)
    elif not isinstance(tags, (list, tuple)):
        tags = [tags]

    # --------------------------
    # initialize discrete states
    # --------------------------
    if init_type == 'random':

        zs = [np.random.choice(K, size=T) for T in Ts]

    elif init_type == 'umap-kmeans':

        import umap
        u = umap.UMAP()
        xs = u.fit_transform(np.vstack(datas))
        km = KMeans(K)
        km.fit(xs)
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

    elif init_type == 'umap-kmeans-diff':

        import umap
        u = umap.UMAP()
        datas_diff = [np.vstack([np.zeros((1, D)), np.diff(data, axis=0)]) for data in datas]
        xs = u.fit_transform(np.vstack(datas_diff))
        km = KMeans(K)
        km.fit(xs)
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

    elif init_type == 'kmeans':

        km = KMeans(K)
        km.fit(np.vstack(datas))
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

    elif init_type == 'kmeans-diff':

        km = KMeans(K)
        datas_diff = [np.vstack([np.zeros((1, D)), np.diff(data, axis=0)]) for data in datas]
        km.fit(np.vstack(datas_diff))
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

    elif init_type == 'kmeans-move':

        D_ = 4
        if datas[0].shape[1] > D_:
            # perform pca
            pca = PCA(D_)
            xs = pca.fit_transform(np.vstack(datas))
            xs = np.split(xs, np.cumsum(Ts)[:-1])
        else:
            # keep original data
            import copy
            D_ = D
            xs = copy.deepcopy(datas)

        model_init = HMM(
            K=2, D=D_, M=0, transitions='standard', observations='ar',
            observations_kwargs={'lags': 1})
        init_model('pca-me', model_init, xs)
        model_init.fit(
            xs, inputs=None, method='em', num_iters=100, tolerance=1e-2,
            initialize=False, transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

        # make still state 0th state
        mses = [np.mean(np.square(model_init.observations.As[i] - np.eye(D_))) for i in range(2)]
        if mses[1] < mses[0]:
            # permute states
            model_init.permute([1, 0])
        moving_state = 1

        inputs_tr = [None] * len(datas)
        zs = [model_init.most_likely_states(x, u) for x, u in zip(xs, inputs_tr)]
        zs = np.concatenate(zs, axis=0)

        # cluster moving data
        km = KMeans(K - 1)
        if np.sum(zs == moving_state) > K - 1:
            datas_diff = [np.vstack([np.zeros((1, D)), np.diff(data, axis=0)]) for data in datas]
            km.fit(np.vstack(datas_diff)[zs == moving_state])
            zs[zs == moving_state] = km.labels_ + 1

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])

    elif init_type == 'ar-clust':

        from sklearn.cluster import SpectralClustering  # , AgglomerativeClustering

        # code from Josh Glaser
        t_win = 5
        t_gap = 5
        num_trials = len(datas)

        if dist_mat is None:
            dist_mat = compute_dist_mat(datas, t_win, t_gap)

        # Cluster!
        clustering = SpectralClustering(n_clusters=K, affinity='precomputed').fit(
            1 / (1 + dist_mat / t_win))

        # Now take the clustered segments, and use them to determine the cluster of the individual
        # time points
        # In the scenario where the segments are nonoverlapping, then we can simply assign the time
        # point cluster as its segment cluster
        # In the scenario where the segments are overlapping, we will let a time point's cluster be
        # the cluster to which the majority of its segments belonged
        # Below zs_init is the assigned discrete states of each time point for a trial. zs_init2
        # tracks the clusters of each time point across all the segments it's part of

        zs = []
        for tr in range(num_trials):
            xhat = datas[tr]
            T = xhat.shape[0]
            n_steps = int((T - t_win) / t_gap) + 1
            t_st = 0
            zs_init = np.zeros(T)
            zs_init2 = np.zeros([T, K])  # For each time point, tracks how many segments it's
            # part of belong to each cluster
            for k in range(n_steps):
                t_end = t_st + t_win
                t_idx = np.arange(t_st, t_end)
                if t_gap == t_win:
                    zs_init[t_idx] = clustering.labels_[k]
                else:
                    zs_init2[t_idx, clustering.labels_[k]] += 1
                t_st = t_st + t_gap
            if t_gap != t_win:
                max_els = zs_init2.max(axis=1)
                for t in range(T):
                    if np.sum(zs_init2[t] == max_els[t]) == 1:
                        # if there's a single best cluster, assign it
                        zs_init[t] = np.where(zs_init2[t] == max_els[t])[0]
                    else:
                        # multiple best clusters
                        if zs_init[t - 1] in np.where(zs_init2[t] == max_els[t])[0]:
                            # use best cluster from previous time point if it's in the running
                            zs_init[t] = zs_init[t - 1]
                        else:
                            # just use first element
                            zs_init[t] = np.where(zs_init2[t] == max_els[t])[0][0]

            # I think this offset is correct rather than just using zs_init, but it should be
            # double checked.
            zs.append(np.concatenate([[0], zs_init[:-1]]))
        zs = np.concatenate(zs, axis=0)

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])

    elif init_type == 'arhmm':

        D_ = 4
        if datas[0].shape[1] > D_:
            # perform pca
            pca = PCA(D_)
            xs = pca.fit_transform(np.vstack(datas))
            xs = np.split(xs, np.cumsum(Ts)[:-1])
        else:
            # keep original data
            import copy
            D_ = D
            xs = copy.deepcopy(datas)

        model_init = HMM(
            K=2, D=D_, M=0, transitions='standard', observations='ar',
            observations_kwargs={'lags': 1})
        init_model('pca-me', model_init, xs)
        model_init.fit(
            xs, inputs=None, method='em', num_iters=100, tolerance=1e-2,
            initialize=False, transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

        # make still state 0th state
        mses = [np.mean(np.square(model_init.observations.As[i] - np.eye(D_))) for i in range(2)]
        if mses[1] < mses[0]:
            # permute states
            model_init.permute([1, 0])
        moving_state = 1

        inputs_tr = [None] * len(datas)
        zs = [model_init.most_likely_states(x, u) for x, u in zip(xs, inputs_tr)]
        zs = np.concatenate(zs, axis=0)

        # cluster moving data
        km = KMeans(K - 1)
        if np.sum(zs == moving_state) > K - 1:
            km.fit(np.vstack(datas)[zs == moving_state])
            zs[zs == moving_state] = km.labels_ + 1

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])

    elif init_type == 'pca-me':

        # pca on motion energy
        datas_filt = np.copy(datas)
        for dtmp in datas_filt:
            for i in range(dtmp.shape[1]):
                dtmp[:, i] = savgol_filter(dtmp[:, i], 5, 2)
        pca = PCA(1)
        me = np.square(np.diff(np.vstack(datas_filt), axis=0))
        xs = pca.fit_transform(np.concatenate([np.zeros((1, D)), me], axis=0))[:, 0]
        xs = xs / np.max(xs)

        # threshold data to get moving/non-moving
        thresh = 0.01
        zs = np.copy(xs)
        zs[xs < thresh] = 0
        zs[xs >= thresh] = 1

        # cluster moving data
        km = KMeans(K - 1)
        km.fit(np.vstack(datas)[zs == 1])
        zs[zs == 1] = km.labels_ + 1

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])

    else:
        raise NotImplementedError('Invalid "init_type" of "%s"' % init_type)

    # ------------------------
    # estimate dynamics params
    # ------------------------
    if init_type != 'em-exact':

        Ezs = [one_hot(z, K) for z in zs]
        expectations = [(Ez, None, None) for Ez in Ezs]

        if str(model.observations.__class__).find('Hierarchical') > -1:
            obs = model.observations
            # initialize parameters for global ar model
            obs.global_ar_model.m_step(expectations, datas, inputs, masks, tags)
            # update prior
            obs._update_hierarchical_prior()
            # Copy global parameters to per-group models
            for ar in obs.per_group_ar_models:
                ar.As = obs.global_ar_model.As.copy()
                ar.Vs = obs.global_ar_model.Vs.copy()
                ar.bs = obs.global_ar_model.bs.copy()
                ar.Sigmas = obs.global_ar_model.Sigmas.copy()

                ar.As = norm.rvs(obs.global_ar_model.As, np.sqrt(obs.cond_variance_A))
                ar.Vs = norm.rvs(obs.global_ar_model.Vs, np.sqrt(obs.cond_variance_V))
                ar.bs = norm.rvs(obs.global_ar_model.bs, np.sqrt(obs.cond_variance_b))
                ar.Sigmas = obs.global_ar_model.Sigmas.copy()
        else:
            model.observations.m_step(expectations, datas, inputs, masks, tags)

    return None


def compute_dist_mat(datas, t_win, t_gap):

    def sse(x, y):
        return np.sum(np.square(x - y))

    from sklearn.linear_model import Ridge

    Ts = [data.shape[0] for data in datas]
    num_trials = len(datas)

    # Elements of segs contain triplets of
    # 1) trial
    # 2) time point of beginning of segment
    # 3) time point of end of segment
    segs = []

    # Get all segments based on predefined t_win and t_gap
    for tr in range(num_trials):
        T = Ts[tr]
        n_steps = int((T - t_win) / t_gap) + 1
        for k in range(n_steps):
            segs.append([tr, k * t_gap, k * t_gap + t_win])

    # Fit a regression (solve for the dynamics matrix) within each segment
    num_segs = len(segs)
    sse_mat = np.zeros([num_segs, num_segs])
    for j, seg in enumerate(segs):
        [tr, t_st, t_end] = seg
        X = datas[tr][t_st:t_end + 1, :]
        rr = Ridge(alpha=1, fit_intercept=True)
        rr.fit(X[:-1], X[1:] - X[:-1])

        # Then see how well the dynamics from segment J works at making predictions on
        # segment K (determined via sum squared error of predictions)
        for k, seg2 in enumerate(segs):
            [tr, t_st, t_end] = seg2
            X = datas[tr][t_st:t_end + 1, :]
            sse_mat[j, k] = sse(X[1:] - X[:-1], rr.predict(X[:-1]))

    # Make "sse_mat" into a proper, symmetric distance matrix for clustering
    tmp = sse_mat - np.diag(sse_mat)
    dist_mat = tmp + tmp.T

    return dist_mat


# -------------------------------------------------------------------------------------------------
# model evaluation functions
# -------------------------------------------------------------------------------------------------


def extract_state_runs(states, indxs, min_length=20):
    """
    Find contiguous chunks of data with the same state

    Args:
        states (list):
        indxs (list):
        min_length (int):

    Returns:
        list
    """

    K = len(np.unique(np.concatenate([np.unique(s) for s in states])))
    state_snippets = [[] for _ in range(K)]

    for curr_states, curr_indxs in zip(states, indxs):
        i_beg = 0
        curr_state = curr_states[i_beg]
        curr_len = 1
        for i in range(1, len(curr_states)):
            next_state = curr_states[i]
            if next_state != curr_state:
                # record indices if state duration long enough
                if curr_len >= min_length:
                    state_snippets[curr_state].append(
                        curr_indxs[i_beg:i])
                i_beg = i
                curr_state = next_state
                curr_len = 1
            else:
                curr_len += 1
        # end of trial cleanup
        if next_state == curr_state:
            # record indices if state duration long enough
            if curr_len >= min_length:
                state_snippets[curr_state].append(curr_indxs[i_beg:i])
    return state_snippets


def viterbi_ll(model, datas):
    """Calculate log-likelihood of viterbi path."""
    inputs = [None] * len(datas)
    masks = [None] * len(datas)
    tags = [None] * len(datas)
    states = [model.most_likely_states(x, u) for x, u in zip(datas, inputs)]
    ll = 0
    for data, input, mask, tag, state in zip(datas, inputs, masks, tags, states):
        if input is None:
            input = np.zeros_like(data)
        if mask is None:
            mask = np.ones_like(data, dtype=bool)
        likelihoods = model.observations.log_likelihoods(data, input, mask, tag)
        ll += np.sum(likelihoods[(np.arange(state.shape[0]), state)])
    return ll


def k_step_ll(model, datas, k_max):
    """Determine the k-step ahead ll."""

    M = (model.M,) if isinstance(model.M, int) else model.M
    L = model.observations.lags  # AR lags

    k_step_lls = 0
    for data in datas:
        input = np.zeros((data.shape[0],) + M)
        mask = np.ones_like(data, dtype=bool)
        pi0 = model.init_state_distn.initial_state_distn
        Ps = model.transitions.transition_matrices(data, input, mask, tag=None)
        lls = model.observations.log_likelihoods(data, input, mask, tag=None)

        T, K = lls.shape

        # Forward pass gets the predicted state at time t given
        # observations up to and including those from time t
        alphas = np.zeros((T, K))
        forward_pass(pi0, Ps, lls, alphas)

        # pz_tt = p(z_{t},  x_{1:t}) = alpha(z_t) / p(x_{1:t})
        pz_tt = np.exp(alphas - logsumexp(alphas, axis=1, keepdims=True))
        log_likes_list = []
        for k in range(k_max + 1):
            if k == 0:
                # p(x_t | x_{1:T}) = \sum_{z_t} p(x_t | z_t) p(z_t | x_{1:t})
                pz_tpkt = np.copy(pz_tt)
                assert np.allclose(np.sum(pz_tpkt, axis=1), 1.0)
                log_likes_0 = logsumexp(lls[k_max:] + np.log(pz_tpkt[k_max:]), axis=1)
            #                 pred_data = get_predicted_obs(model, data, pz_tpkt)
            else:
                if k == 1:
                    # p(z_{t+1} | x_{1:t}) =
                    # \sum_{z_t} p(z_{t+1} | z_t) alpha(z_t) / p(x_{1:t})
                    pz_tpkt = np.copy(pz_tt)

                # p(z_{t+k} | x_{1:t}) =
                # \sum_{z_{t+k-1}} p(z_{t+k} | z_{t+k-1}) p(z_{z+k-1} | x_{1:t})
                if Ps.shape[0] == 1:  # stationary transition matrix
                    pz_tpkt = np.matmul(pz_tpkt[:-1, None, :], Ps)[:, 0, :]
                else:  # dynamic transition matrix
                    pz_tpkt = np.matmul(pz_tpkt[:-1, None, :], Ps[k - 1:])[:, 0, :]
                assert np.allclose(np.sum(pz_tpkt, axis=1), 1.0)

                # p(x_{t+k} | x_{1:t}) =
                # \sum_{z_{t+k}} p(x_{t+k} | z_{t+k}) p(z_{t+k} | x_{1:t})
                log_likes = logsumexp(lls[k:] + np.log(pz_tpkt), axis=1)
                # compute summed ll only over timepoints that are valid for each value of k
                log_likes_0 = log_likes[k_max - k:]

            log_likes_list.append(np.sum(log_likes_0))

    k_step_lls += np.array(log_likes_list)

    return k_step_lls


def k_step_r2(
        model, datas, k_max, n_samp=10, obs_noise=True, disc_noise=True, return_type='total_r2'):
    """Determine the k-step ahead r2.

    Args:
        model:
        datas:
        k_max:
        n_samp:
        obs_noise: bool
            turn observation noise on/off
        disc_noise: bool
            turn discrete state sampling on/off
        return_type:
            'per_batch_r2'
            'total_r2'
            'bootstrap_r2'
            'per_batch_mse'

    Returns:

    """

    N = len(datas)
    L = model.observations.lags  # AR lags
    D = model.D

    x_true_total = []
    x_pred_total = [[] for _ in range(k_max)]
    if return_type == 'per_batch_r2':
        k_step_r2s = np.zeros((N, k_max, n_samp))
    elif return_type == 'total_r2':
        k_step_r2s = np.zeros((k_max, n_samp))
    else:
        raise NotImplementedError('"%s" is not a valid return type' % return_type)

    for d, data in enumerate(datas):
        # print('%i/%i' % (d + 1, len(datas)))

        T = data.shape[0]

        x_true_all = data[L + k_max - 1: T + 1]
        x_pred_all = np.zeros((n_samp, (T - 1), D, k_max))

        if not disc_noise:
            zs = model.most_likely_states(data)
            inputs = np.zeros((T,) + (model.observations.M,))

        # collect sampled data
        for t in range(L - 1, T):
            # find the most likely discrete state at time t based on its past
            if disc_noise:
                data_t = data[:t + 1]
                zs = model.most_likely_states(data_t)[-L:]
            else:
                pass

            # sample forward in time n_samp times
            for n in range(n_samp):
                # sample forward in time k_max steps
                if disc_noise:
                    _, x_pred = model.sample(
                        k_max, prefix=(zs, data_t[-L:]), with_noise=obs_noise)
                else:
                    pad = L
                    x_pred = np.concatenate((data[t - L + 1:t + 1], np.zeros((k_max, D))))
                    for k in range(pad, pad + k_max):
                        if t + 1 + k - pad < T:
                            x_pred[k, :] = model.observations.sample_x(
                                zs[t + 1 + k - pad], x_pred[:k], input=inputs[t], tag=None,
                                with_noise=obs_noise)
                        else:
                            # beyond the end of the data sample; return zeros
                            pass
                    x_pred = x_pred[pad:]

                # predicted x values in the forward prediction time
                x_pred_all[n, t - L + 1, :, :] = np.transpose(x_pred)[None, None, :, :]

        # store predicted data
        x_true_total.append(x_true_all)
        for k in range(k_max):
            idxs = (k_max - k - 1, k_max - k - 1 + x_true_all.shape[0])
            x_pred_total[k].append(x_pred_all[:, slice(*idxs), :, k])

    # compute r2s
    if return_type == 'per_batch_r2':
        for d in range(len(datas)):
            for k in range(k_max):
                for n in range(n_samp):
                    k_step_r2s[d, k, n] = r2_score(
                        x_true_total[d], x_pred_total[k][d][n])

    elif return_type == 'total_r2':
        for k in range(k_max):
            for n in range(n_samp):
                k_step_r2s[k, n] = r2_score(
                    np.vstack(x_true_total),
                    np.vstack([x_pred_total[k][d][n] for d in range(len(datas))]))

    return k_step_r2s


# -------------------------------------------------------------------------------------------------
# path handling functions
# -------------------------------------------------------------------------------------------------

def get_model_name(n_states, model_kwargs):
    trans = model_kwargs['transitions']
    obs = model_kwargs['observations']
    if obs.find('ar') > -1:
        lags = model_kwargs['observation_kwargs']['lags']
    else:
        lags = 0
    if trans == 'sticky':
        kappa = model_kwargs['transition_kwargs']['kappa']
    else:
        kappa = ''
    model_name = str(
        'obs=%s_trans=%s_lags=%i_K=%02i' % (obs, trans, lags, n_states))
    if trans == 'sticky':
        model_name = str('%s_kappa=%1.0e' % (model_name, kappa))
    return model_name


def plot_latents_states(
        latents=None, states=None, state_probs=None, slc=(0, 1000), m=20):
    """
    states | state probs | x coords | y coords

    Args:
        latents (dict): keys are 'x', 'y', 'l', each value is a TxD np array
        states (np array): length T
        state_probs (np array): T x K
    """

    n_dlc_comp = latents.shape[1]

    if state_probs is not None:
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10),
            gridspec_kw={'height_ratios': [0.1, 0.1, 0.4]})
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 10),
            gridspec_kw={'height_ratios': [0.1, 0.4]})

    i = 0
    axes[i].imshow(states[None, slice(*slc)], aspect='auto', cmap='tab20b')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title('State')

    # if state_probs is not None:
    #     i += 1
    #     n_states = state_probs.shape[1]
    #     xs_ = [np.arange(slc[0], slc[1]) for _ in range(n_states)]
    #     ys_ = [state_probs[slice(*slc), j] for j in range(n_states)]
    #     cs_ = [j for j in range(n_states)]
    #     _multiline(xs_, ys_, ax=axes[i], c=cs_, alpha=0.8, cmap='tab20b', lw=3)
    #     axes[i].set_xticks([])
    #     axes[i].set_xlim(slc[0], slc[1])
    #     axes[i].set_yticks([])
    #     axes[i].set_ylim(-0.1, 1.1)
    #     axes[i].set_title('State probabilities')

    i += 1
    behavior = m * latents / np.max(np.abs(latents)) + \
        np.arange(latents.shape[1])
    axes[i].plot(np.arange(slc[0], slc[1]), behavior[slice(*slc), :])
    axes[i].set_xticks([])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(0, n_dlc_comp + 1)

    axes[-1].set_xlabel('Time (bins)')
    plt.tight_layout()
    plt.show()

    return fig

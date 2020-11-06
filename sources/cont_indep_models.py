import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from sources.derive_preds import derive_preds



def load_models(cont_indep_model_names, cont_indep_model_filenames, include_concat, verbose):
    """"Load and process the context-independent models.

    The context-independent models (i.e., embeddings) are read from file, which is assumed to have no header.
    The first column in each file must contains the words, while the other columns must contain the vector
    dimensions.

    Parameters
    ----------
    cont_indep_model_names : str array, shape (n_cont_indep_models)
        Names of the context-independent models, where n_cont_indep_models is the number of models.

    cont_indep_model_filenames : str array, shape (n_cont_indep_models)
        Names of the files storing the context-independent models (i.e., embeddings), where
        n_cont_indep_models is the number of models.

    include_concat : bool
        Whether to automatically generate and include the concatenation of all the context-independent
        models.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    cont_indep_models : DataFrame array, shape ({n_cont_indep_models, n_cont_indep_models + 1})
        Context-independent models, where n_cont_indep_models is the number of context-independent models.
        Each model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the
        number of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.
    """

    cont_indep_models = []

    # iterate through all the models
    for curr_cont_indep_model_filename, curr_cont_indep_model_name in zip(cont_indep_model_filenames,
                                                                          cont_indep_model_names):

        if verbose:

            start_time = time()

        # load each model from file
        curr_cont_indep_model = pd.read_csv(curr_cont_indep_model_filename, header=None)

        # cast the vectors to half precision, in order to save memory and computation time
        curr_cont_indep_model.iloc[:,1:] = curr_cont_indep_model.iloc[:,1:].astype(np.float16)


        # generate column names for the current model
        curr_cont_indep_model_col_names = ['Word']

        for i in range(len(curr_cont_indep_model.columns)-1):

            curr_cont_indep_model_col_names.append(curr_cont_indep_model_name + '_Dim_' + str(i+1))

        curr_cont_indep_model.columns = curr_cont_indep_model_col_names

        # save each loaded model
        cont_indep_models.append(curr_cont_indep_model)

        if verbose:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Loaded {} model'.format(run_duration, curr_cont_indep_model_name))


    if include_concat:

        if verbose:

            start_time = time()

        # concatenate all the models
        curr_cont_indep_model = cont_indep_models[0]

        for i in range(len(cont_indep_models)-1):

            curr_cont_indep_model = curr_cont_indep_model.merge(cont_indep_models[i+1], on='Word')

        # generate column names for the concatenated model
        curr_cont_indep_model_col_names = ['Word']

        for i in range(len(curr_cont_indep_model.columns)-1):

            curr_cont_indep_model_col_names.append('Cont_Indep_Models_Concat_Dim_' + str(i+1))

        curr_cont_indep_model.columns = curr_cont_indep_model_col_names

        # save the concatenated model, under the name of 'Cont_Indep_Models_Concat'
        cont_indep_model_names.append('Cont_Indep_Models_Concat')
        cont_indep_models.append(curr_cont_indep_model)

        if verbose:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Loaded {} model'.format(run_duration, 'Cont_Indep_Models_Concat'))

    return cont_indep_models



def reduce_dims(cont_indep_models, cont_indep_model_names, include_concat, n_pcs, verbose):
    """Reduce the dimensionality of (a subset of) context-independent models.

    The dimensionality reduction is performed using Principal Component Analysis (PCA). When deriving
    model-based predictors, the selected models are replaced with their reduced versions.

    Parameters
    ----------
    cont_indep_models : DataFrame array, shape ({n_cont_indep_models, n_cont_indep_models + 1})
        Individual context-independent models, where n_cont_indep_models is the number of individual
        context-independent models. Each model has shape (n_words, n_dims+1), where n_words is the number of
        words, and n_dims is the number of vector dimensions, both of which are specific to each model. For
        each model, the first column ('Word') contains the words, while the other columns contain the vector
        dimensions.

    cont_indep_model_names : str array, shape (n_indiv_cont_indep_models)
        Names of the individual context-independent models, where n_indiv_cont_indep_models is the number of
        individual models.

    include_concat : bool
        Whether to include the concatenated model.

    n_pcs : int
        Number of principal components per model.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    None.
    """

    print('Reducing dimensionality...')

    # if necessary, include concatenated model
    if include_concat:

        cont_indep_model_names.append('Cont_Indep_Models_Concat')

    # iterate through all the models
    for curr_cont_indep_model_name in cont_indep_model_names:

        if verbose:

            start_time = time()

        curr_cont_indep_model_pos = cont_indep_model_names.index(curr_cont_indep_model_name)

        curr_cont_indep_model = cont_indep_models[curr_cont_indep_model_pos]

        curr_cont_indep_model_PCA = PCA(n_components=n_pcs)

        # run PCA with the specified number of components and generate column names for the reduced models
        curr_dim_red_cont_indep_model = pd.concat([curr_cont_indep_model[['Word']],
                                                   pd.DataFrame(curr_cont_indep_model_PCA.fit_transform(
                                                       curr_cont_indep_model.iloc[:,1:]),
                                                       columns=[curr_cont_indep_model_name + '_PC_' + str(i+1)
                                                                for i in range(n_pcs)])],
                                                  axis=1)

        # cast the vectors to half precision, in order to save memory and computation time
        curr_dim_red_cont_indep_model.iloc[:,1:] = curr_dim_red_cont_indep_model.iloc[:,1:].astype(np.float16)

        # replace the selected original models with their reduced versions
        cont_indep_models[curr_cont_indep_model_pos] = curr_dim_red_cont_indep_model

        if verbose:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Reduced dimensionality of {} model'.format(run_duration, curr_cont_indep_model_name))



def generate_preds(stimuli, cont_indep_models, cont_indep_model_names, pred_names, use_orig_stimuli,
                   target_is_inflected, verbose):
    """Generate predictors from the context-independent models.

    The predictors are derived from the previously loaded models.

    Parameters
    ----------
    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    cont_indep_models : DataFrame array, shape ({n_cont_indep_models, n_cont_indep_models + 1})
        Context-independent models, where n_cont_indep_models is the number of context-independent models.
        Each model has shape (n_words, n_dims+1), where n_words is the number of words, and n_dims is the
        number of vector dimensions, both of which are specific to each model. For each model, the first
        column ('Word') contains the words, while the other columns contain the vector dimensions.

    cont_indep_model_names : str array, shape (n_cont_indep_models)
        Names of the individual context-independent models, where n_cont_indep_models is the number of
        individual models.

    pred_names : str array, shape (n_norms_and_models_sel)
        Names of the behavioural norms and distributional models selected by the user, where 
        n_norms_and_models_sel is the number of selected norms and models.

    use_orig_stimuli : bool array, shape (n_norms_and_models_sel)
        When deriving predictors, whether to use the original stimuli or the translated ones, where
        n_norms_and_models_sel is the number of selected norms and models.

    target_is_inflected : bool array, shape (n_cont_indep_models_sel)
        For each set of predictors, whether to use the inflected form of the target (i.e., the word from
        position INDEX in TEXT), or the uninflected one (i.e., the word from TARGET), where
        n_cont_indep_models_sel is the number of selected models.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    preds_cont_indep_models : DataFrame array, shape (n_cont_indep_models_sel)
        Predictors derived from the context-independent models selected by the user, where
        n_cont_indep_models_sel is the number of such models. Each set of predictors is of shape
        (n_stimuli, n_preds), where n_stimuli is the number of words, and n_preds is the number of predictors.
    """
    
    preds_cont_indep_models = []

    if len(cont_indep_models) > 0:

        # iterate through all the selected models
        for curr_pred_name, curr_use_orig_stimuli, curr_target_is_inflected in zip(pred_names,
                                                                                   use_orig_stimuli,
                                                                                   target_is_inflected):

            if curr_pred_name in cont_indep_model_names:
                
                if verbose:

                    start_time = time()
                
                curr_cont_indep_model = cont_indep_models[cont_indep_model_names.index(curr_pred_name)]

                # check whether to derive predictors for the original stimuli or the translated ones, and
                # whether to use the inflected or uninflected form of the target
                if curr_use_orig_stimuli:

                    # derive predictors for the original stimuli, using the current model
                    curr_preds = derive_preds(stimuli['X'], curr_cont_indep_model, curr_pred_name,
                                              curr_target_is_inflected)

                elif curr_target_is_inflected:

                    # derive predictors for the translated stimuli, based on the inflected form of the
                    # target, using the current model
                    curr_preds = derive_preds(stimuli['X_tr_infl'], curr_cont_indep_model, curr_pred_name,
                                              False)

                else:

                    # derive predictors for the translated stimuli, based on the uninflected form of the
                    # target, using the current model
                    curr_preds = derive_preds(stimuli['X_tr_uninfl'], curr_cont_indep_model, curr_pred_name,
                                              False)

                # derive density predictors, based on the current model
                curr_density_preds = compute_density(curr_preds, curr_pred_name, curr_cont_indep_model)

                preds_cont_indep_models.append(pd.concat([curr_preds, curr_density_preds], axis=1))
                
                if verbose:
        
                    finish_time = time()

                    run_duration = int(finish_time - start_time)

                    # notify the user of the successful completion of the task, together with its duration
                    if curr_target_is_inflected:

                        print('({}s) Generated predictors for {} - inflected target'.format(run_duration,
                                                                                            curr_pred_name))

                    else:

                        print('({}s) Generated predictors for {} - uninflected target'.format(run_duration,
                                                                                              curr_pred_name))

    return preds_cont_indep_models



def compute_density(curr_preds, curr_cont_indep_model_name, curr_cont_indep_model):
    """Generate density predictors from the context-independent models.

    Parameters
    ----------
    curr_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors (i.e., unreduced/reduced vector representations) derived from the context-independent
        model, where n_stimuli is the number of stimuli, n_preds is the number of predictors.

    curr_cont_indep_model_name : str
        Name of the context-independent model based on which the density predictors are derived.

    curr_cont_indep_model : DataFrame array, shape (n_words, n_dims + 1)
        Context-independent model, where n_words is the number of words, and n_dims is the number of vector
        dimensions, both of which are specific to each model. The first column ('Word') contains the words,
        while the other columns contain the vector dimensions.

    Returns
    -------
    preds_density : DataFrame, shape (n_stimuli, 4)
        Predictors derived from the context-independent model, where n_stimuli is the number of stimuli.
    """

    # number of closest neighbours over which the density is computed
    n_neigh_density = 20

    n_dims = curr_cont_indep_model.shape[1] - 1


    # compute the density predictor for the target
    cos_sim_target = np.matmul(normalize(curr_preds.values[:, :n_dims]),
                               normalize(curr_cont_indep_model.values[:,1:]).transpose())
    preds_density_target = np.mean(np.sort(cos_sim_target)[:,-n_neigh_density:], axis=1, keepdims=True)

    # compute the density predictor for the context
    cos_sim_context = np.matmul(normalize(curr_preds.values[:, n_dims:2*n_dims]),
                                normalize(curr_cont_indep_model.values[:,1:]).transpose())
    preds_density_context = np.mean(np.sort(cos_sim_context)[:,-n_neigh_density:], axis=1, keepdims=True)

    # compute the absolute difference between the density predictors for the target and context
    preds_density_abs_diff = np.abs(preds_density_target - preds_density_context)

    # compute the product between the density predictors for the target and context
    preds_density_prod = preds_density_target * preds_density_context


    preds_density = pd.DataFrame(np.hstack([preds_density_target, preds_density_context,
                                            preds_density_abs_diff, preds_density_prod]))
    preds_density.columns = [dens_type + '_Dens_' + curr_cont_indep_model_name for dens_type in ('Target',
                                                                                                 'Context',
                                                                                                 'Abs_Diff',
                                                                                                 'Prod')]
    
    preds_density = preds_density.reindex(index=np.zeros([preds_density.shape[0]], dtype=np.int))

    return preds_density



import numpy as np
import pandas as pd
from time import time

from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from transformers import BartConfig, BartTokenizer, BartModel
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from sources.misc_utils import search_sequence_numpy



def load_models(cont_dep_model_names, cont_dep_model_ids, verbose):
    """Load and process the context-dependent models.

    Parameters
    ----------
    cont_dep_model_names : str array, shape (n_cont_dep_models)
        Names of the context-dependent models, where n_cont_dep_models is the number of models. The only
        model names (i.e., classes of models) currently supported by our implementation are 'albert',
        'alberto', 'bart', 'bert', 'gpt-2', and 'roberta'.

    cont_dep_model_ids : str array, shape (n_cont_dep_models)
        Ids of pre-trained Hugging Face models, where n_cont_dep_models is the number of models. Most classes
        of models consist of more than one model (e.g., in the case of BERT, valid ids are
        'bert-base-uncased', 'bert-large-cased', 'bert-base-multilingual-uncased', etc.).

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    tokenizers : PreTrainedTokenizer array, shape (n_cont_dep_models)
        Tokenizers corresponding to the context-dependent models, where n_cont_dep_models is the number of
        context-dependent models. The context-dependent models and the tokenizers are matched position-wise.

    cont_dep_models : PreTrainedModel array, shape (n_cont_dep_models)
        Context-dependent models, where n_cont_dep_models is the number of context-dependent models.
    """

    tokenizers = []
    cont_dep_models = []

    # iterate through all the models
    for curr_cont_dep_model_name, curr_cont_dep_model_id in zip(cont_dep_model_names, cont_dep_model_ids):

        if verbose:

            start_time = time()

        # select the loader appropriate for each model
        if curr_cont_dep_model_name.lower() == 'albert':

            load_cont_dep_model_func = load_Albert

        elif curr_cont_dep_model_name.lower() == 'alberto':

            load_cont_dep_model_func = load_Alberto

        elif curr_cont_dep_model_name.lower() == 'bart':

            load_cont_dep_model_func = load_Bart

        elif curr_cont_dep_model_name.lower() == 'bert':

            load_cont_dep_model_func = load_Bert

        elif curr_cont_dep_model_name.lower() == 'gpt-2':

            load_cont_dep_model_func = load_Gpt2

        elif curr_cont_dep_model_name.lower() == 'roberta':

            load_cont_dep_model_func = load_Roberta

        else:

            print('ERROR: {} model cannot be loaded!'.format(curr_cont_dep_model_name))

            load_cont_dep_model_func = None;

        # load and save each (tokenizer, model) pair
        curr_tokenizer_and_model = load_cont_dep_model_func(curr_cont_dep_model_id)

        tokenizers.append(curr_tokenizer_and_model[0])
        cont_dep_models.append(curr_tokenizer_and_model[1])

        if verbose and load_cont_dep_model_func != None:

            finish_time = time()

            run_duration = int(finish_time - start_time)

            # notify the user of the successful completion of the task, together with its duration
            print('({}s) Loaded {} model'.format(run_duration, curr_cont_dep_model_name))

    return tokenizers, cont_dep_models



def generate_preds(stimuli, tokenizers, cont_dep_models, cont_dep_model_names, pred_names, use_orig_stimuli,
                   target_is_inflected, verbose):
    """Generate predictors from the context-dependent models.

    The predictors are derived from the previously loaded models.

    Parameters
    ----------
    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    tokenizers : PreTrainedTokenizer array, shape (n_cont_dep_models)
        Tokenizers corresponding to the context-dependent models, where n_cont_dep_models is the number of
        context-dependent models. The context-dependent models and the tokenizers are matched position-wise.

    cont_dep_models : PreTrainedModel array, shape (n_cont_dep_models)
        Context-dependent models, where n_cont_dep_models is the number of context-dependent models.

    cont_dep_model_names : str array, shape (n_cont_dep_models)
        Names of the context-dependent models, where n_cont_dep_models is the number of models. The only
        model names (i.e., classes of models) currently supported by our implementation are 'albert',
        'alberto', 'bart', 'bert', 'gpt-2', and 'roberta'.

    pred_names : str array, shape (n_norms_and_models_sel)
        Names of the behavioural norms and distributional models selected by the user, where 
        n_norms_and_models_sel is the number of selected norms and models.

    use_orig_stimuli : bool array, shape (n_norms_and_models_sel)
        When deriving predictors, whether to use the original stimuli or the translated ones, where
        n_norms_and_models_sel is the number of selected norms and models.

    target_is_inflected : bool array, shape (n_cont_dep_models_sel)
        For each set of predictors, whether to use the inflected form of the target (i.e., the word from
        position INDEX in TEXT), or the uninflected one (i.e., the word from TARGET), where
        n_cont_dep_models_sel is the number of selected models.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    preds_cont_dep_models : DataFrame array, shape (n_cont_dep_models_sel)
        Predictors derived from the context-dependent models selected by the user, where n_cont_dep_models_sel
        is the number of such models. Each set of predictors is of shape (n_words, n_preds), where n_words is
        the number of words, and n_preds is the number of predictors.
    """
    
    preds_cont_dep_models = []

    if len(cont_dep_models) > 0:

        # iterate through all the selected models
        for curr_pred_name, curr_use_orig_stimuli, curr_target_is_inflected in zip(pred_names,
                                                                                   use_orig_stimuli,
                                                                                   target_is_inflected):

            if curr_pred_name in cont_dep_model_names:

                if verbose:

                    start_time = time()

                # select appropriate (tokenizer, model) pair
                curr_tokenizer = tokenizers[cont_dep_model_names.index(curr_pred_name)]
                curr_cont_dep_model = cont_dep_models[cont_dep_model_names.index(curr_pred_name)]

                if curr_pred_name.lower() == 'albert':

                    generate_preds_func = generate_preds_Albert

                elif curr_pred_name.lower() == 'alberto':

                    generate_preds_func = generate_preds_Alberto

                elif curr_pred_name.lower() == 'bart':

                    generate_preds_func = generate_preds_Bart

                elif curr_pred_name.lower() == 'bert':

                    generate_preds_func = generate_preds_Bert

                elif curr_pred_name.lower() == 'gpt-2':

                    generate_preds_func = generate_preds_Gpt2

                elif curr_pred_name.lower() == 'roberta':

                    generate_preds_func = generate_preds_Roberta

                else:

                    print('ERROR: It is not possible to generate predictors for the {} model!'.format(
                        curr_pred_name))

                    generate_preds_func = None

                # check whether to derive predictors for the original stimuli or the translated ones, and
                # whether to use the inflected or uninflected form of the target
                if curr_use_orig_stimuli:

                    # derive predictors for the original stimuli, using the current model
                    curr_preds = generate_preds_func(curr_tokenizer, curr_cont_dep_model, stimuli['X'],
                                                           curr_target_is_inflected)

                elif curr_target_is_inflected:

                    # derive predictors for the translated stimuli, based on the inflected form of the
                    # target, using the current model
                    curr_preds = generate_preds_func(curr_tokenizer, curr_cont_dep_model,
                                                           stimuli['X_tr_infl'], False)

                else:

                    # derive predictors for the translated stimuli, based on the uninflected form of the
                    # target, using the current model
                    curr_preds = generate_preds_func(curr_tokenizer, curr_cont_dep_model,
                                                           stimuli['X_tr_uninfl'], False)

                # generate column names for the predictors
                curr_preds.columns = [curr_pred_name + '_' + str(i) for i in range(curr_preds.shape[1])]

                preds_cont_dep_models.append(curr_preds)

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

    return preds_cont_dep_models



def load_Albert(cont_dep_model_id):
    """Load Albert tokenizer and model.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Albert class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Albert model.

    model : PreTrainedModel
        Selected Albert model.
    """

    configuration = AlbertConfig.from_pretrained(cont_dep_model_id, output_hidden_states=True)
    tokenizer = AlbertTokenizer.from_pretrained(cont_dep_model_id)
    model = AlbertModel.from_pretrained(cont_dep_model_id, config=configuration)

    return tokenizer, model



def load_Alberto(cont_dep_model_id):
    """Load Alberto tokenizer and model.

    Since Alberto is a Bert model trained over an Italian corpus, it uses the same tokenizer and model as Bert.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Alberto class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Alberto model.

    model : PreTrainedModel
        Selected Alberto model.
    """

    tokenizer = BertTokenizer.from_pretrained(cont_dep_model_id)
    model = BertModel.from_pretrained(cont_dep_model_id, output_hidden_states=True)

    return tokenizer, model



def load_Bart(cont_dep_model_id):
    """Load Bart tokenizer and model.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Bart class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Bart model.

    model : PreTrainedModel
        Selected Bart model.
    """

    configuration = BartConfig.from_pretrained(cont_dep_model_id, output_hidden_states=True)
    tokenizer = BartTokenizer.from_pretrained(cont_dep_model_id, add_prefix_space=True)
    model = BartModel.from_pretrained(cont_dep_model_id, config=configuration)

    return tokenizer, model



def load_Bert(cont_dep_model_id):
    """Load Bert tokenizer and model.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Bert class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Bert model.

    model : PreTrainedModel
        Selected Bert model.
    """

    configuration = BertConfig.from_pretrained(cont_dep_model_id, output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained(cont_dep_model_id)
    model = BertModel.from_pretrained(cont_dep_model_id, config=configuration)

    return tokenizer, model



def load_Gpt2(cont_dep_model_id):
    """Load Gpt-2 tokenizer and model.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Gpt-2 class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Gpt-2 model.

    model : PreTrainedModel
        Selected Gpt-2 model.
    """

    configuration = GPT2Config.from_pretrained(cont_dep_model_id, output_hidden_states=True)
    tokenizer = GPT2Tokenizer.from_pretrained(cont_dep_model_id, add_prefix_space=True)
    model = GPT2Model.from_pretrained(cont_dep_model_id, config=configuration)

    return tokenizer, model



def load_Roberta(cont_dep_model_id):
    """Load Roberta tokenizer and model.

    Parameters
    ----------
    cont_dep_model_id : str
        Id of pre-trained Hugging Face model, belonging to the Roberts class of models.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Roberta model.

    model : PreTrainedModel
        Selected Roberta model.
    """

    configuration = RobertaConfig.from_pretrained(cont_dep_model_id, output_hidden_states=True)
    tokenizer = RobertaTokenizer.from_pretrained(cont_dep_model_id)
    model = RobertaModel.from_pretrained(cont_dep_model_id, config=configuration)

    return tokenizer, model



def generate_preds_Albert(tokenizer, model, stimuli, target_is_inflected):
    """Generate predictors from the Albert model.

    The predictors consist of the activations in the last hidden layer, associated with the target, when the
    target is presented in context.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Albert model.

    model : PreTrainedModel
        Selected Albert model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Albert model selected by the user, where n_stimuli is the number of
        stimuli, and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        if target_is_inflected:

            target_token_raw = tokenizer.encode(curr_target_context.split(' ')[curr_target_position],
                                                return_tensors='pt')

        else:

            target_token_raw = tokenizer.encode(curr_target, return_tensors='pt')

        target_token_final = target_token_raw.detach().numpy()[0][1:-1]

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[-1]

        if len(target_indices) > 0:

            curr_preds = np.mean(hidden_states[-1].detach().numpy()[:, target_indices, :], axis=1)

        else:

            curr_preds = np.mean(hidden_states[-1].detach().numpy(), axis=1)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



def generate_preds_Alberto(tokenizer, model, stimuli, target_is_inflected, use_last_four_layers=True):
    """Generate predictors from the Alberto model.

    The predictors consist of the activations in the last hidden layer (or averaged over the last four layers),
    associated with the target, when the target is presented in context.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Alberto model.

    model : PreTrainedModel
        Selected Alberto model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Alberto model selected by the user, where n_stimuli is the number of
        stimuli, and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        if target_is_inflected:

            target_token_raw = tokenizer.encode(curr_target_context.split(' ')[curr_target_position],
                                                return_tensors='pt')

        else:

            target_token_raw = tokenizer.encode(curr_target, return_tensors='pt')

        target_token_final = target_token_raw.detach().numpy()[0][1:-1]

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[2]

        if len(target_indices) > 0:

            if use_last_four_layers:

                curr_preds = np.mean(np.vstack([curr_hidden_states.detach().numpy()[:, target_indices, :]
                                                 for curr_hidden_states in hidden_states[-4:]]),
                                      axis=(0,1))[np.newaxis,:]

            else:

                curr_preds = np.mean(hidden_states[-1].detach().numpy()[:, target_indices, :], axis=1)

        else:

            if use_last_four_layers:

                curr_preds = np.mean(np.vstack([curr_hidden_states.detach().numpy()
                                                 for curr_hidden_states in hidden_states[-4:]]),
                                      axis=(0,1))[np.newaxis,:]

            else:

                curr_preds = np.mean(hidden_states[-1].detach().numpy(), axis=1)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



def generate_preds_Bart(tokenizer, model, stimuli, target_is_inflected):
    """Generate predictors from the Bart model.

    The predictors consist of the activations in the last hidden layer, associated with the target, when the
    target is presented in context.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Bart model.

    model : PreTrainedModel
        Selected Bart model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Bart model selected by the user, where n_stimuli is the number of stimuli,
        and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        if target_is_inflected:

            target_token_raw = tokenizer.encode(curr_target_context.split(' ')[curr_target_position],
                                                return_tensors='pt')

        else:

            target_token_raw = tokenizer.encode(curr_target, return_tensors='pt')

        target_token_final = target_token_raw.detach().numpy()[0][1:-1]

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[2]

        if len(target_indices) > 0:

            curr_preds = np.expand_dims(np.mean(hidden_states[-1].detach().numpy()[target_indices, :],
                                                 axis=0),
                                         axis=0)

        else:

            curr_preds = np.expand_dims(np.mean(hidden_states[-1].detach().numpy(), axis=0), axis=0)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



def generate_preds_Bert(tokenizer, model, stimuli, target_is_inflected, use_last_four_layers=True):
    """Generate predictors from the Bert model.

    The stimuli are derived from the activations in the last hidden layer, or from the activations averaged
    over the last four hidden layers.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Bert model.

    model : PreTrainedModel
        Selected Bert model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Bert model selected by the user, where n_stimuli is the number of stimuli,
        and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        if target_is_inflected:

            target_token_raw = tokenizer.encode(curr_target_context.split(' ')[curr_target_position],
                                                return_tensors='pt')

        else:

            target_token_raw = tokenizer.encode(curr_target, return_tensors='pt')

        target_token_final = target_token_raw.detach().numpy()[0][1:-1]

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[2]

        if len(target_indices) > 0:

            if use_last_four_layers:

                curr_preds = np.mean(np.vstack([curr_hidden_states.detach().numpy()[:, target_indices, :]
                                                 for curr_hidden_states in hidden_states[-4:]]),
                                      axis=(0,1))[np.newaxis,:]

            else:

                curr_preds = np.mean(hidden_states[-1].detach().numpy()[:, target_indices, :], axis=1)

        else:

            if use_last_four_layers:

                curr_preds = np.mean(np.vstack([curr_hidden_states.detach().numpy()
                                                 for curr_hidden_states in hidden_states[-4:]]),
                                      axis=(0,1))[np.newaxis,:]

            else:

                curr_preds = np.mean(hidden_states[-1].detach().numpy(), axis=1)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



def generate_preds_Gpt2(tokenizer, model, stimuli, target_is_inflected):
    """Generate predictors from the Gpt-2 model.

    The predictors consist of the activations in the last hidden layer, associated with the target, when the
    target is presented in context.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Gpt-2 model.

    model : PreTrainedModel
        Selected Gpt-2 model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Gpt-2 model selected by the user, where n_stimuli is the number of stimuli,
        and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        if target_is_inflected:

            target_token_raw = tokenizer.encode(curr_target_context.split(' ')[curr_target_position],
                                                return_tensors='pt')

        else:

            target_token_raw = tokenizer.encode(curr_target, return_tensors='pt')

        target_token_final = target_token_raw.detach().numpy()[0]

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[2]

        if len(target_indices) > 0:

            curr_preds = np.mean(hidden_states[-1].detach().numpy()[:, target_indices, :], axis=1)

        else:

            curr_preds = np.mean(hidden_states[-1].detach().numpy(), axis=1)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



def generate_preds_Roberta(tokenizer, model, stimuli, target_is_inflected):
    """Generate predictors from the Roberta model.

    The predictors consist of the activations in the last hidden layer, associated with the target, when the
    target is presented in context.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer corresponding to the selected Roberta model.

    model : PreTrainedModel
        Selected Roberta model.

    stimuli : DataFrame, shape (n_stimuli, n_col)
        Experimental stimuli, following the format used the organizers, where n_stimuli is the number of
        stimuli, and n_col is the number of columns. The dataset must include at least the columns TARGET,
        INDEX, and TEXT.

    target_is_inflected : bool
        Whether to use the inflected form of the target (i.e., the word from position INDEX in TEXT), or the
        uninflected one (i.e., the word from TARGET).

    Returns
    -------
    cont_preds : DataFrame, shape (n_stimuli, n_preds)
        Predictors derived from the Roberta model selected by the user, where n_stimuli is the number of
        stimuli, and n_preds is the number of predictors (i.e., the size of the last hidden layer).
    """

    cont_preds = pd.DataFrame()

    targets = list(stimuli['TARGET'])
    target_contexts = list(stimuli['TEXT'])
    target_positions = np.array(stimuli['INDEX'])

    # iterate over the stimuli
    for curr_target, curr_target_context, curr_target_position in zip(targets, target_contexts,
                                                                      target_positions):

        # generate target encoding
        encoding_dict = {}

        encoding_dict[curr_target_context.split(' ')[0]] = tokenizer.encode(curr_target_context.split(' ')[0],
                                                                            add_special_tokens=False,
                                                                            add_prefix_space=False)

        for curr_word in curr_target_context.split()[1:]:

            encoding_dict[curr_word] = tokenizer.encode(curr_word, add_special_tokens=False,
                                                        add_prefix_space=True)

        if curr_target_context.split(' ')[0] != curr_target:

            encoding_dict[curr_target] = tokenizer.encode(curr_target, add_special_tokens=False,
                                                          add_prefix_space=True)

        else:

            encoding_dict[curr_target] = tokenizer.encode(curr_target, add_special_tokens=False,
                                                          add_prefix_space=False)

        if target_is_inflected:

            target_token_final = np.array(encoding_dict[curr_target_context.split(' ')[curr_target_position]][0])

        else:

            target_token_final = np.array(encoding_dict[curr_target][0])

        # generate context encoding
        input_ids = tokenizer.encode(curr_target_context, return_tensors='pt')

        input_ids_array = input_ids.detach().numpy()[0]

        # locate and retrieve the predictors
        target_indices = search_sequence_numpy(input_ids_array, target_token_final)

        hidden_states = model(input_ids)[-1]

        if len(target_indices) > 0:

            curr_preds = np.mean(hidden_states[-1].detach().numpy()[:, target_indices, :], axis=1)

        else:

            curr_preds = np.mean(hidden_states[-1].detach().numpy(), axis=1)

        cont_preds = pd.concat([cont_preds, pd.DataFrame(curr_preds)])

    return cont_preds



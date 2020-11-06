import pandas as pd
from time import time

from transformers import MarianTokenizer, MarianMTModel



def generate_English_translation(stimuli, verbose):
    """Translate the Italian stimuli into English.

    The translation is performed using the MarianMT system.

    Parameters
    ----------
    stimuli : stimuli : DataFrame, shape (n_stimuli, n_cols_stimuli)
        Experimental stimuli in Italian, following the format used the organizers, where n_stimuli is the
        number of stimuli, and n_cols_stimuli is the number of columns. The dataset must include at least the
        columns TARGET, POS, INDEX, and TEXT.

    verbose : bool
        Whether to inform the user of the successful completion of the task, together with its duration.

    Returns
    -------
    translated_stimuli_infl : DataFrame, shape (n_stimuli, 4)
        English translation of the Italian stimuli, using the inflected form of the target, where n_stimuli
        is the number of stimuli. The four columns correspond to the original columns TARGET, POS, INDEX, and
        TEXT.

    translated_stimuli_uninfl : DataFrame, shape (n_stimuli, 4)
        English translation of the Italian stimuli, using the uninflected form of the target, where n_stimuli
        is the number of stimuli. The four columns correspond to the original columns TARGET, POS, INDEX, and
        TEXT.
    """

    print('Translating Italian stimuli...')

    if verbose:

        start_time = time()

    # load MarianMT tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-it-en")


    translated_targets_infl = []
    translated_targets_uninfl = []
    translated_texts = []

    # iterate over the stimuli
    for curr_stimulus_position in range(stimuli.shape[0]):

        curr_index = stimuli['INDEX'][curr_stimulus_position]
        curr_text = stimuli['TEXT'][curr_stimulus_position]

        curr_target_infl = curr_text.split(' ')[int(curr_index)]
        curr_target_uninfl = stimuli['TARGET'][curr_stimulus_position]

        # prepare the stimuli for translation
        translation = [model.generate(**tokenizer.prepare_seq2seq_batch([curr_target_infl])),
                       model.generate(**tokenizer.prepare_seq2seq_batch([curr_target_uninfl])),
                       model.generate(**tokenizer.prepare_seq2seq_batch([curr_text]))]

        # translate and collect the stimuli
        translated_target_infl = tokenizer.batch_decode(translation[0], skip_special_tokens=True)[0]
        translated_target_uninfl = tokenizer.batch_decode(translation[1], skip_special_tokens=True)[0]
        translated_text = tokenizer.batch_decode(translation[2], skip_special_tokens=True)[0]

        translated_targets_infl.append(translated_target_infl)
        translated_targets_uninfl.append(translated_target_uninfl)
        translated_texts.append(translated_text)


    translated_stimuli_infl = pd.DataFrame(list(zip(translated_targets_infl, stimuli['POS'],
                                                    stimuli['INDEX'], translated_texts)), columns=['TARGET',
                                                                                                   'POS',
                                                                                                   'INDEX',
                                                                                                   'TEXT'])

    translated_stimuli_uninfl = pd.DataFrame(list(zip(translated_targets_uninfl, stimuli['POS'],
                                                      stimuli['INDEX'], translated_texts)), columns=['TARGET',
                                                                                                     'POS',
                                                                                                     'INDEX',
                                                                                                     'TEXT'])

    if verbose:

        finish_time = time()

        run_duration = int(finish_time - start_time)

        # notify the user of the successful completion of the task, together with its duration
        print('({}s) Translated stimuli from Italian to English'.format(run_duration))

    return translated_stimuli_infl, translated_stimuli_uninfl



# Predicting concreteness in context for English and Italian by using distributional models and behavioural norms

We provide an implementation of the models with which we participated (as team Andi) in the CONcreTEXT task of EVALITA 2020. The task involved predicting subjective ratings of concreteness for words presented in context. Our approach, which ranked first in both the English and Italian tracks, relies on a combination of context-dependent and context-independent distributional models, together with behavioural norms.

If you want to test our models, you can run the English and Italian demos (see the two Jupyter notebooks). Please feel free to experiment with your own combinations of stimuli, norms, and models, once you make sure that they are in the proper format (see the information provided below). If you get interesting results, please let us know! 🙂  

## Before you start
  
In order to be able to successfully run the demos, you first need to do the following things:

1) **Create a dedicated Python environment (highly recommended) and install the necessary libraries**. Start by [installing pytorch](https://pytorch.org/get-started/locally/). Next, run the following command:

```bash
pip install notebook pandas scipy scikit-learn transformers
```

2) Place the necessary files in their corresponding directories, as follows:

* **Put the files 'CONcreTEXT_trial_EN.tsv' and 'CONcreTEXT_trial_IT.tsv' in the 'stimuli' folder.** 
The two files can be obtained from the organizers, visit the [CONcreTEXT 2020 site](https://lablita.github.io/CONcreTEXT/) for more details.

* **(Optional) Put the behavioural norms in the 'behavioural-norms' folder.** Each file must be in .CSV format and have a header with the variable names (e.g., Word,Frequency,SemanticDiversity,[...]). The first column ('Word') must contain the normed words, while the other columns must contain the behavioural data. For copyright reasons, we cannot upload the norms we used for our submission, but you can download them yourself by using the following links (just remember to convert them to the right format, keeping only the columns of interest):
    * [concreteness](http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt)
    * [semantic diversity](https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0278-x/MediaObjects/13428_2012_278_MOESM1_ESM.xlsx)  
    * [frequency and contextual diversity](http://crr.ugent.be/papers/SUBTLEX-UK.xlsx)      
    * [age of acquisition](http://crr.ugent.be/papers/AoA_ratings_Kuperman_et_al_BRM.zip)
    * [emotional dimensions](https://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon-Aug2018Release.zip)
    * [sensorimotor dimensions](https://osf.io/48wsc/download)
    * [other (potentially useful) norms](http://crr.ugent.be/programs-data/megastudy-data-available) 
* **(Optional) Put the context-independent embeddings (i.e., models) in the 'context-independent-models' folder.** Each file must be in .CSV format, but with no header. The first column must contain the words, while the other columns must contain the word vectors. The models we used for our submission can be downloaded from the dedicated [OSF project](https://osf.io/px2gm/).

3) **(Optional) Make sure you have enough disk space for the context-dependent models (i.e., Hugging Face transformers).** If you wish to change the location where the models are stored, uncomment the first two lines in the demo code and replace \<new_cache_folder_path> with your chosen location. If you decide to use such models, keep in mind that it might take some time for the download, given that the size of most models is around 500MB.

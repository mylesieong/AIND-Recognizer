import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    probabilities = []
    guesses = []

    # Calculate Probabilities
    word_ids = test_set.get_all_sequences().keys() # e.g. [0,1,2,3]
    for wid in word_ids:
        recog_dict = {}
        for model_word in models.keys(): # e.g. ['FISH', 'JOHN', 'BREAD']
            model = models[model_word]
            item_X, item_lengths = test_set.get_item_Xlengths(wid)
            try:
                recog_dict[model_word] = model.score(item_X, item_lengths)
            except:
                recog_dict[model_word] = float("-inf")
        probabilities.append(recog_dict)

    # Calculate guesses
    for wid in word_ids:
        guesses.append(max(probabilities[wid]))

    for wid in range(0,5):
        print("For sequence {}, the probabilities dict is {} and the guess list is {}".format(test_set.get_item_sequences(wid), probabilities[wid], guesses[wid]))

    # return probabilities, guesses
    return (probabilities, guesses)

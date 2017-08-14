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
    print("asl_single_data: get_all_sequences:{}".format(test_set.get_all_sequences()))
    print("asl_single_data: get_all_Xlengths:{}".format(test_set.get_all_Xlengths()))
    print("asl_single_data: get_item_sequences:{}".format(test_set.get_item_sequences(0)))
    print("asl_single_data: get_item_Xlengths:{}".format(test_set.get_item_Xlengths(0)))
    probabilities = []
    guesses = []

    # Calculate Probabilities
    word_ids = test_set.get_all_sequences().keys() # e.g. [0,1,2,3]
    for wid in word_ids:
        recog_dict = {}
        for model_word in models.keys(): # e.g. ['FISH', 'JOHN', 'BREAD']
            model = models[model_word]
            recog_dict[model_word] = model.score(test_set.get_item_Xlengths(wid))
        probabilities.append(recog_dict)

    # return probabilities, guesses
    return (None, None)

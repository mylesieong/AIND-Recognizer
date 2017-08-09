import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Build model candidates list
        modelcandidates = []
        for n in range(self.min_n_components, self.max_n_components):
            model = self.base_model(n)
            bic = -2 * model.score(self.X) + n * math.log(self.lengths)  
            modelcandidates.append( (bic, model) )

        # Select the best model in BIC context
        (_, bestmodel) = min(modelcandidates) 

        return bestmodel

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Build model candidates list
        modelcandidates = []
        for n in range(self.min_n_components, self.max_n_components):
            model = self.base_model(n)
            modelcandidates.append(model)

        # Calc summary of all log likelihood
        scoresum = 0
        for m in modelcandidates:
            scoresum += m.score(self.X)

        # Calc the DIC for each model candidate and put into a dictationary
        modeldic = {}
        for m in modelcandidates:
            modeldic[m] = m.score(self.X) - 1/(len(modelcandidates) - 1)*(scoresum - m.score(self.X))
        
        # Select the best model in DIC context
        (_, bestmodel) = min(modeldic) 
        
        ## TODO Verify min/max

        return bestmodel

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):

        def get_model_cv(self, num_of_state):
            total, index = 0
            for cv_train_idx, cv_test_idx in KFold().split(self.all_word_sequences):
                hmm_model = GaussianHMM(n_components=num_of_state, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(cv_train_idx_X, cv_train_idx_lengths)
                total = total + hmm_model.score(cv_test_idx_X, cv_test_idx_lengths)
                index = index + 1

            return total/index 

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Build array which element is (average_likelihood, model)
        modelcandidates = []
        for n in range(self.min_n_components, self.max_n_components):
            model = self.base_model(n)
            modelcandidates.append( (get_model_cv(n), model) )

        (_, bestmodel) = max(modelcandidates)

        return bestmodel

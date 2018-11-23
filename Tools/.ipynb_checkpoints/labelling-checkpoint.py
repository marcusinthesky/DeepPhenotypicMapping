from time import time
import os

import pandas as pd
import numpy as np  

from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.compose import TransformedTargetRegressor


def pseudo_fit(self, X, y=None, **fit_params):
    """Fit to data, then transform it.
    Adds pseudo-labelling method to any Scikit-Learn Estimator Class
    ----------
    X : numpy array of shape [n_samples, n_features]
        Training set.
    y : numpy array of shape [n_samples]
        Target values.
    Returns
    -------
    X_new : numpy array of shape [n_samples, n_features_new]
        Transformed array.
    """
    if (y==-1).sum()>0:
        self.fit(X[y!=-1], y[y!=-1], **fit_params)
        y[y==-1] = self.predict(X[y==-1], **fit_params)
    
        #sample_weights[y==-1] = sample_weights[y!=-1].median()
    
    return self.fit(X, y, **fit_params)


class SemiSup_RandomizedSearchCV(BaseEstimator):
    def __init__(self, estimator, param_distributions, n_iter=100, cv=5, scoring=metrics.accuracy_score, pseudo=True):
        # We initialize our class similar to sklearn randomized search
        self.estimator = estimator
        self.scoring = scoring
        self.pseudo = pseudo
        
        self.transformedtargetestimator = TransformedTargetRegressor(regressor=estimator,
                                                    func=lambda x: x if np.random.rand() > 1/cv else -1, 
                                                    inverse_func=lambda x: x, check_inverse=False)
        self.scoring = scoring
        self.sampler = ParameterSampler(param_distributions, n_iter)
        self.cv_results_ = pd.DataFrame({'mean_test_score': np.empty(shape=[0]),
                                         'std_test_score': np.empty(shape=[0]),
                                         'mean_score_time': np.empty(shape=[0]),
                                         'std_score_time': np.empty(shape=[0]),
                                         'params': None})
        self.folds = KFold(n_splits=cv)
        
    def fit(self, X, y, sample_weight=None):
        for params in self.sampler:
            # Update Parameters
            self.estimator.set_params(**params)
            # Reset Scores
            scores = []
            times = []
            
            for train_index, test_index in self.folds.split(X):
                #Create Semisupervised Sampler
                self.transformedtargetestimator = TransformedTargetRegressor(regressor=self.estimator,
                                                                             func=lambda x: np.where(np.in1d(x.index,train_index),x,-1), 
                                                                             inverse_func=lambda x: x, check_inverse=False)
                #Fit
                if self.pseudo:
                    self.transformedtargetestimator.regressor.pseudo_fit = pseudo_fit.__get__(self.transformedtargetestimator.regressor)
                    self.transformedtargetestimator = self.transformedtargetestimator.regressor.pseudo_fit(X, self.transformedtargetestimator.func(y))
                else:
                    self.transformedtargetestimator.fit(X, y, sample_weight)
                    
                #Score
                score_index = np.in1d(y.index,test_index)
                start = time()
                scores.append(self.scoring(y[score_index], self.transformedtargetestimator.predict(X=X[score_index])))
                times.append(time()-start)
            self.cv_results_ = self.cv_results_.append(pd.DataFrame({'mean_test_score': np.mean(scores),
                                                                     'std_test_score': np.std(scores),
                                                                     'mean_score_time': np.mean(times),
                                                                     'std_score_time': np.std(times),
                                                                     'params': [params]}))
        self.cv_results_ = self.cv_results_.sort_values('mean_test_score', ascending=False).reset_index(drop=True)
        return self
        
# write out results to csv and serialized feature format
def write_out(dataframe, name, parent_dir=False, result=False, scores=True, feather=True):
    if scores:
        results = dataframe\
                    .loc[:,['mean_test_score','std_test_score','mean_score_time','std_score_time', 'params']]\
                    .sort_values('mean_test_score', ascending=False)\
                    .reset_index(drop=True)

        results = pd.concat([results.drop(columns=['params']),
                                results.params.astype(str)],axis=1)
    
        results['model'] = name
    else:
        results = dataframe.reset_index(drop=True)
    
    path = os.path.join('.','Data','Models' if not parent_dir else '',f'{name}')
    results.to_csv(path+'.csv')
    if feather:
        results.to_feather(path+'.feather')
    
    if result==True:
        return results

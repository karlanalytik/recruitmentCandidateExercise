"""Model related tools"""
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

def cross_validation_table(X, y: np.array, model = LinearRegression(), 
                            cross_val = TimeSeriesSplit()):
    """Generates dataframe with the results obtained in each fold during cross 
    validation process.

    Parameters
    ----------
    X: dataframe
        The dataframe whose columns are to be standardized
    y: array
        An array with sales series (target)
    model: sklearn model, optional
        sklearn model, default = LinearRegression()
    cross_val: sklearn object, optional
        Sample split technique
    
    Returns
    -------
    dataframe
        A dataframe with the results obtained during cross validation
    """
    cv_df = pd.DataFrame(columns = ['fold', 'X_train_index', 'X_train_dates', 
                        'X_test_index', 'X_test_dates'])
    for i, (train_index, test_index) in enumerate(cross_val.split(X)):
        fold = i
        train_ind_init = str(train_index[0])
        train_ind_fin = str(train_index[-1])
        test_ind_init = str(test_index[0])
        test_ind_fin = str(test_index[-1])
        train_dates_init = str(X.index[train_index[0]]).replace('00:00:00', '')
        train_dates_fin = str(X.index[train_index[-1]]).replace('00:00:00', '')
        test_dates_init = str(X.index[test_index[0]]).replace('00:00:00', '')
        test_dates_fin = str(X.index[test_index[-1]]).replace('00:00:00', '')
        new_row = pd.Series({'fold': fold, 
                    'X_train_index': train_ind_init + ' - ' + train_ind_fin, 
                    'X_train_dates': train_dates_init + ' - ' + train_dates_fin, 
                    'X_test_index': test_ind_init + ' - ' + test_ind_fin, 
                    'X_test_dates': test_dates_init + ' - ' + test_dates_fin})
        cv_df = pd.concat([cv_df, new_row.to_frame().T], ignore_index = True)        
        cv_results = cross_validate(model, X, y, cv = cross_val, 
                                scoring = ('r2', 'neg_mean_squared_error'))
        df = cv_df.join(pd.DataFrame(cv_results))
    return df




def find_hyperparams(X, y, model = Lasso(), cross_val = TimeSeriesSplit()):
    param_grid = {
        "alpha": [x for x in range(1, 100)] + [y/10 for y in range(10)],
        "tol": [0.00001, 0.0000001, 0.01],
        "selection": ['cyclic', 'random']
        }
    clf = GridSearchCV(model, param_grid, cv = cross_val, error_score = -1000, 
    n_jobs = -1, scoring = 'r2')
    clf.fit(X, y)
    print("Best score: " + str(clf.best_score_))
    summary = pd.DataFrame(clf.cv_results_)
    summary.sort_values(by = 'rank_test_score')
    dc_scores = {}
    dc_scores_esc = {}
    dc_scores[str(model).split('(')[0]] = {'model': clf.best_estimator_, 
    'score': clf.best_score_}
    return dc_scores, dc_scores_esc, summary
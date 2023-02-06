from modeltools import cross_validation_table
from sklearn.linear_model import LinearRegression
from formattools import col_formatting, campagin_to_col
from sklearn.model_selection import TimeSeriesSplit
from mediaadtools import CarryOverEffect
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


class MediaModel():
    """...

    ...

    Attributes
    ----------
    strength: float
        A float indicating how much gets carried over
    lenght: int
        An integer indicating how long does it get carried over
    a: int
        A non-negative integer used to simulate saturation 
    
    Methods
    -------
    prepare_data()
        Prepares the data to be able to apply the models
    get_crossval_table()
        Create a cross validation summary and measures means
    make_fit_plot()
        Generates chart showing the model fit
    predict()
        Generates prediction for the data
    make_fit_plot()
        Generates chart showing the model fit
    contribution_plot()
        Generates graphs showing the adjusted contribution of each campaign
    """


    def __init__(self, strength: float, lenght: int, a: int):
        """
        Parameters
        ----------
        strength: float
            A float indicating how much gets carried over
        lenght: int
            An integer indicating how long does it get carried over
        a: int
            A non-negative integer used to simulate saturation 
        """

        self.strength = strength
        self.lenght = lenght
        self.a = a      


    def prepare_data(self):
        """Prepares the data to be able to apply the models.

        Prepares the data to be consumed by the model. Standardize column names,
        separate campaigns into individual variables, apply transformations to 
        expenditures (carry over effect and saturation).

        Parameters
        ----------
        

        Raises
        ------
        dataframe
            A dataframe with original and transformed columns
        X
            A dataframe with transformed columns (predictive variables)
        y
            An array with sales series (target)
        """
        #infile = 'data.csv'
        infile = Path(__file__).parent / 'data.csv'
        df = pd.read_csv(infile, 
                        parse_dates = ['Date (Week)'],
                        index_col = 'Date (Week)')
        df.columns = col_formatting(df)
        df = campagin_to_col(df, 'media_campaign', 'media_spend_usd')
        # Carry Over Effect transformation
        for col in ['camp_1_spend', 'camp_2_spend', 'camp_3_spend']:
            spend_col = col + '_adstock'
            df[spend_col] = CarryOverEffect(df[col], self.strength, 
                                        self.lenght).convolve()
        X = df.drop(columns = ['search_volume', 'camp_1', 'camp_2', 'camp_3', 
                                'camp_1_spend', 'camp_2_spend', 'camp_3_spend'])
        y = df['search_volume']
        # Saturation transformation
        # Applies an exponential function to transform the spending series and 
        # to simulate saturation. It's monotonically increasing and it takes 
        # values between (0, 1].
        #X = 1 - np.exp(-self.a * X)
        return df, X, y


    #def find_hyperparm(self, X, y, model):
        """...

        ...

        Parameters
        ----------
        

        Raises
        ------
        ...
            ...
        ...
            ...
        ...
            ...
        """
        
        #return df, X, y

    
    def get_crossval_table(self, X, y, model = LinearRegression()):
        """Create a cross validation summary and measures means.
        
        Generates dataframe with the results obtained in each fold during 
        cross validation process. It calculates the k-fold  and k-1 fold means.

        Parameters
        ----------
        X: dataframe
            A dataframe with transformed columns (predictive variables)
        y: array
            An array with sales series (target)
        model: sklearn model, optional
            sklearn model, default = LinearRegression()

        Raises
        ------
        cv_df: dataframe
            A dataframe containing cross validation summary
        cv_r2_mean: float
            A float indicating the mean of the k-folds (r2)
        cv_n_1_r2_mean: float
            A float indicating the mean of the k-1-folds (r2)
        """
        # We do not use the standard k-fold cross-validation here because we are
        # dealing with time series data.
        ts_cv = TimeSeriesSplit(n_splits = 4)
        cv_df = cross_validation_table(X, y, model, ts_cv)
        cv_r2_mean = cv_df['test_r2'].mean()
        cv_n_1_r2_mean = cv_df.iloc[:-1]['test_r2'].mean()
        return cv_df, cv_r2_mean, cv_n_1_r2_mean


    def predict(self, X, y):
        """Generates prediction for the data.
        
        Generates the prediction based in X and y, the model is 
        LinearRegression().

        Parameters
        ----------
        X: dataframe
            A dataframe with transformed columns (predictive variables)
        y: array
            An array with sales series (target)
        model: sklearn model, optional
            sklearn model, default = LinearRegression()

        Raises
        ------
        coef: array
            Float array containing the regression coefficients
        intercept: float
            A float corresponding to the intercept of the regression
        y_pred: float
            An array with predicted sales series
        """
        lr = LinearRegression()
        lr.fit(X, y) # refit the model with the complete dataset
        coef = lr.coef_
        intercept = lr.intercept_
        y_pred = lr.predict(X)
        return coef, intercept, y_pred

    
    def make_fit_plot(self, X, y, y_pred):
        """Generates chart showing the model fit.

        Plots the observed values together with the values obtained by the 
        model.

        Parameters
        ----------
        X: dataframe
            A dataframe with transformed columns (predictive variables)
        y: array
            An array with sales series (target)
        y_pred: array
            An array with predicted sales series

        Raises
        ------
        fig
            A chart showing the model fit
        """
        fig, ax = plt.subplots(figsize = (16, 10), layout = 'constrained')
        ax.plot(X.index, y, label='Observed') 
        ax.plot(X.index, y_pred, label='Predicted') 
        ax.set_xlabel('Week') 
        ax.set_ylabel('Google Search volumes') 
        ax.set_title('Model fitting (Historical data vs. Predictions)') 
        ax.legend()
        return fig

    def calculate_roi(self, X, y, coef, intercept):
        weights = pd.Series(coef, index = X.columns)
        base = intercept
        unadj_contributions = X.mul(weights).assign(Base = base)
        adj_contributions = (unadj_contributions
                            .div(unadj_contributions.sum(axis = 1), axis = 0)
                            .mul(y, axis = 0)
                            ) # contains all contributions for each day
        return adj_contributions

    def roi_table(self, df, adj_contributions):
        roi_df = pd.DataFrame(columns = ['campaign_n', 'searches_from_camp_n', 'spendings_on_camp_n', 'roi'])
        for i in range(1, 4):
            col1 = 'camp_'+ str(i) +'_spend_adstock'
            col2 = 'camp_'+ str(i) +'_spend'
            new_row = pd.Series({'campaign_n': i, 
                                'searches_from_camp_n': adj_contributions[col1].sum(),
                                'spendings_on_camp_n': df[col2].sum(), 
                                'roi': adj_contributions[col1].sum() / df[col2].sum()})
            roi_df = pd.concat([roi_df, new_row.to_frame().T], ignore_index = True)
            roi_df['1/roi'] = 1 / roi_df['roi']
        return roi_df
    
    def contribution_plot(self, adj_contributions):
        """Generates graphs showing the adjusted contribution of each campaign.

        Graph the contribution of each campaign to the search volume for the 
        period. Displays the campaigns and the base in different colors.

        Parameters
        ----------
        adj_contributions: dataframe
            A dataframe with the adjusted contributions for each campaign 

        Raises
        ------
        fig
            A chart showing adjusted contributions and base
        """
        ax = (adj_contributions[['Base', 'camp_1_spend_adstock', 
                                'camp_2_spend_adstock', 'camp_3_spend_adstock']]
        .plot.area(
            figsize = (16, 10),
            linewidth = 1,
            title = 'Predicted Searches and Breakdown',
            ylabel = 'Sales',
            xlabel = 'Date')
                )
        ax.set_xlabel('Week') 
        ax.set_ylabel('Google Search volumes') 
        ax.set_title('Actual vs Pred') 
        ax.legend()
        return plt.figure()
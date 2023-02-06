"""Format and display functions"""

import pandas as pd

def col_formatting(df):
    """Removes special characters from column names.

    Parameters
    ----------
    df: dataframe
        The dataframe whose columns are to be standardized
    
    Returns
    -------
    list
        A list of standarized column names
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    new_cols = []
    for i in range(0, df.shape[1]):
        new_col = ''.join([a for a in df.columns[i] if a.isalnum() or a == '_'])
        new_cols.append(new_col)
    return new_cols

def campagin_to_col(df, camp_col: str, media_spend_col: str):
    """Generates individual columns for campaigns.

    Parameters
    ----------
    df: dataframe
        The dataframe whose columns are to be separated
    camp_col: str
        The name of the column containing the campaigns
    media_spend_col:
        The name of the column including the spending series
    
    Returns
    -------
    dataframe
        A dataframe in which each column corresponds to a single campaign
    """

    df = df.join(pd.get_dummies(df[camp_col].apply(str), prefix = 'camp'))
    for col in [col for col in df.columns if col.startswith('camp_')]:
        spend_col = col + '_spend'
        df[spend_col] = df[col] * df[media_spend_col]
    df.drop([camp_col, media_spend_col], axis = 1, inplace = True)
    return df
from .base import *
import warnings
import numpy as np


# ------------------------------------------------------------------------------
# `head` and `tail`
# ------------------------------------------------------------------------------

@dfpipe
def head(df, n=5):
    return df.head(n)


@dfpipe
def tail(df, n=5):
    return df.tail(n)


# ------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------


@dfpipe
def sample(df, *args, **kwargs):
    return df.sample(*args, **kwargs)


@pipe
@group_delegation
@symbolic_evaluation(eval_as_label=['*'])
def distinct(df, *args, **kwargs):
    if not args:
        return df.drop_duplicates(**kwargs)
    return df.drop_duplicates(list(args), **kwargs)

@pipe
@group_delegation
@symbolic_evaluation(eval_as_label=['*'])
def shape(df):
    return df.shape


@dfpipe
def row_slice(df, indices):
    if isinstance(indices, (tuple, list)):
        indices = np.array(indices)
    if isinstance(indices, int):
        indices = np.array([indices])
    if isinstance(indices, pd.Series):
        indices = indices.values

    if indices.dtype == bool:
        return df.loc[indices, :]
    else:
        return df.iloc[indices, :]


# ------------------------------------------------------------------------------
# Filtering/masking
# ------------------------------------------------------------------------------

@dfpipe
def mask(df, *args):
    mask = pd.Series(np.ones(df.shape[0], dtype=bool))
    for arg in args:
        if arg.dtype != bool:
            raise Exception("Arguments must be boolean.")
        mask = mask & arg.reset_index(drop=True)
    return df[mask.values]

filter_by = mask   # alias for mask()


# Helper function for mask or filter_by with multiple columns
def mult(col: str, items: list):
    arg_list = []
    string = all(isinstance(x, str) for x in items)
    numeric = all(isinstance(x, (int,float)) for x in items)
    if not any([string,numeric]):
        raise TypeError("all items must be either string or numeric")
    for item in items:
        if string:
            arg_list.append(f"(X.{col} == '{item}')")
        elif numeric:
            arg_list.append(f"(X.{col} == {item})")
    arg_list = '|'.join(arg_list)
    return eval(arg_list)




@dfpipe
def top_n(df, n=None, ascending=True, col=None):
    if not n:
        raise ValueError('n must be specified')
    if not isinstance(col, pd.Series):
        col = df.columns[-1]
    else:
        col = col._name
    index = df[[col]].copy()
    index['ranks'] = index[col].rank(ascending=ascending)
    index = index[index['ranks'] >= index['ranks'].nlargest(n).min()]
    return df.reindex(index.index)


@dfpipe
def pull(df, column=-1):
    return df.ix[:, column]

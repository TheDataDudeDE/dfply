from .base import *


@pipe
@group_delegation
@symbolic_evaluation(eval_as_label=['*'])
def fillna(df, **kwargs):
    return df.fillna(**kwargs)
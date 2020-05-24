from .base import *



@pipe
@group_delegation
@symbolic_evaluation(eval_as_label=['*'])
def plot_scatter(df, x, y, s=None,c=None , **kwargs):
    return df.plot.scatter(x=x,y=y,s=s,c=c, **kwargs)
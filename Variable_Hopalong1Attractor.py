#=================================================================================
#---------------    Variable Value and Hopalong1 Attractor    --------------------
#=================================================================================
#                                _____________
#------------------     X = Y - √|(b * X - c)| * sign(X)   -----------------------
#------------------     Y = a - X                          -----------------------

#=================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf
from colorcet import palette_n

#---------------------------------------------------------------------------------

ps = {k:p[::-1] for k, p in palette_n.items()}

pn.extension()

#---------------------------------------------------------------------------------

@jit(nopython=True)
def hopalong1_trajectory(a, b, c, x0, y0, n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):
        x[i+1] = y[i] - np.sqrt(abs(b * x[i] - c)) * np.sign(x[i])
        y[i+1] = a - x[i]
        
    return x, y

#---------------------------------------------------------------------------------

def hopalong1_plot(a=1.000, b=2.000, c=6.600, n=4000000, colormap=ps['fire']):
    
    cvs = ds.Canvas(plot_width=500, plot_height=500)
    x, y = hopalong1_trajectory(a, b, c, 0, 0, n)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap=colormap)

#---------------------------------------------------------------------------------

pn.interact(hopalong1_plot, n=(1,10000000), colormap=ps)

#---------------------------------------------------------------------------------

# The value of this attractor can be changed freely.
# Try it in the jupyter notebook.


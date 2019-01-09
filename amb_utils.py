import numpy as np
import bpy # pylint: disable=import-error
import bmesh # pylint: disable=import-error
import random

import cProfile, pstats, io



def profiling_start():
    # profiling
    pr = cProfile.Profile()
    pr.enable()
    return pr

def profiling_end(pr):
    # end profile, print results
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats(20)
    print(s.getvalue())


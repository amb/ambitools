import cProfile
import pstats
import io


# TODO: with aut.set_mode("OBJECT")  # OBJECT, EDIT ...


def profiling_start():
    # profiling
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiling_end(pr):
    # end profile, print results
    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s)
    ps.strip_dirs().sort_stats(sortby).print_stats(20)
    print(s.getvalue())

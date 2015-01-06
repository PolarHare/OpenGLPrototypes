from contextlib import contextmanager
import timeit

def current_time_ms():
    return timeit.default_timer() * 1000

@contextmanager
def timer(task_name = None):
    start = current_time_ms()
    yield
    if task_name is None:
        print "Time passed: {} ms".format(current_time_ms() - start)
    else:
        print "Time passed for {}: {} ms".format(task_name, current_time_ms() - start)
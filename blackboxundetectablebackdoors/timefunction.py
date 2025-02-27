import time

def timefunction(fun):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fun(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper

from time import time
import logging 

def timer(func): 
    # This function shows the execution time of  
    # the function object passed 

    log = logging.getLogger("nexarAPI.timer")
    log.setLevel(logging.INFO)

    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        log.debug(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 
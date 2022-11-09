import sys
from queue import LifoQueue
from typing import Generator, Callable, Any


def safe_recursion_bootstrap(func: Callable[[Any], Generator]):
    """
    stack runs like decorator -> func -> decorator -> return, so the recursion stack never exceeds 2.

    Usage: put a stack in your kwargs and it maintains the recursion.
    """

    def wrapped_func(*args, **kwargs):
        stack: LifoQueue = kwargs['stack']
        if not stack.empty():               # where next goes to, return the generator directly
            return func(*args, **kwargs)
        recur = func(*args, **kwargs)       # initialized generator
        while True:
            if isinstance(recur, Generator):
                stack.put(recur)
                recur = next(recur)
            else:       # it's a number then, computation done for this branch
                stack.get()
                if stack.empty():
                    break
                recur = stack.queue[-1].send(recur)     # send the result back to its parent
        return recur

    return wrapped_func


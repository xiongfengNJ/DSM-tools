"""
This module holds all the ungrouped utilities.
"""

from queue import LifoQueue
from typing import Generator, Callable, Any


def safe_recursion_bootstrap(func: Callable[[Any], Generator]) -> Callable[[Any], Any]:
    """Convert a recursion function to a stack-size-limitless one, without changing the recursion limit of your system.

    It requires that:
    1. your recursion function has been refactored to a **generator**, i.e. `yield` instead of `return` everywhere,
    even for `return None`, which is often omitted at the end of the function. Also, `yield` for any call of itself
    within the recursion.
    2. The decorator adds a new keyword argument 'stack' to the recursion function, so your recursion function should
    not contain a 'stack' argument itself.
    3. Everytime you start the recursion, you must provide a new stack by keyword 'stack', as something
    like `func(.., stack=LifoQueue())`, and within the function, the signature must be `func(..., stack=stack)` to
    pass this stack along. Don't do anything else with the stack. The decorator will handle everything.

    This way, you can call the recursion and retrieve the result same as the normal one.

    ### Explanations

    To be short, it turns the searching process into a generator lets you initialize a stack to store the generators.
    Long as there is memory for the stack, there will be no overflow. The speed is comparable to the bare recursion.

    The reason for passing a new stack each time you call is for the compatibility for parallel computation. A decorator
    with a stack in there is fine only when the decoration takes place locally.

    The recursion runs like decorator -> func -> decorator -> return, so the recursion stack never exceeds 2.
    """

    def wrapped_func(*args, stack: LifoQueue, **kwargs):
        if not stack.empty():               # where next goes to, return the generator directly
            return func(*args, stack=stack, **kwargs)
        recur = func(*args, stack=stack, **kwargs)       # initialized generator
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


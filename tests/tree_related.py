import pytest
from multiprocessing import Pool
from queue import LifoQueue
import traceback
from functools import cache


def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = stack.append(f(*args, **kwargs))
            while True:
                try:
                    to = stack.append(stack[-1].send(to))
                except StopIteration as e:
                    stack.pop()
                    to = e.value
                    if not stack:
                        break
            return to

    return wrappedfunc


# @cache
@bootstrap
def fun(n):
    if n == 0:
        return 1
    return (yield fun(n - 1)) + 1


if __name__ == '__main__':
    print(fun(1000000))
    print(fun(1000000))
    print(fun(1000000))
    print(fun(1000000))
    print(fun(1000000))
    print(fun(1000000))

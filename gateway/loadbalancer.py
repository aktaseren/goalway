from itertools import cycle

_balancers = {}

def get_backend(model, backends):
    if model not in _balancers:
        _balancers[model] = cycle(backends)
    return next(_balancers[model])
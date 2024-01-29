import os   

DEFAULT_ROOT_PATH = './materials'

datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = DEFAULT_ROOT_PATH
    return datasets[name](**kwargs)
    
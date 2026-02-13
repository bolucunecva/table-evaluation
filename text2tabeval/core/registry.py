# core/registry.py
METRICS = {}

def register_metric(cls):
    METRICS[cls.name] = cls
    return cls
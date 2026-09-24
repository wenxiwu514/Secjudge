from .api import API


def get_api_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == 'hf':
        from .hf_api import HFAPI
        return HFAPI
    else:
        raise ValueError(f'Unknown API name {name}')


__all__ = ['get_api_class_from_name', 'API','openai_completions']

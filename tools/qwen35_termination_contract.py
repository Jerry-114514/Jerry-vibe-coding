"""Candidate termination fix; not enabled in the frozen v4 execution path."""


def explicit_eos_ids(tokenizer, generation_config):
    """Include tokenizer's assistant EOS and preserve configured sequence EOS."""
    native = tokenizer.eos_token_id
    if type(native) is not int or native < 0:
        raise ValueError('Tokenizer must define a valid EOS token id')
    configured = generation_config.eos_token_id
    values = configured if isinstance(configured, (list, tuple)) else [configured]
    values = [native] + [value for value in values if value is not None]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('Invalid generation EOS token id')
    return list(dict.fromkeys(values))

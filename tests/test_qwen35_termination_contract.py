from types import SimpleNamespace

import pytest

from tools.qwen35_termination_contract import explicit_eos_ids


def test_tokenizer_assistant_eos_is_included_with_model_fallback():
    assert explicit_eos_ids(SimpleNamespace(eos_token_id=248046),
                            SimpleNamespace(eos_token_id=248044)) == [248046, 248044]


def test_configured_list_is_preserved_without_duplicates():
    assert explicit_eos_ids(SimpleNamespace(eos_token_id=248046),
                            SimpleNamespace(eos_token_id=[248044, 248046])) == [248046, 248044]


def test_missing_configured_eos_uses_tokenizer():
    assert explicit_eos_ids(SimpleNamespace(eos_token_id=248046),
                            SimpleNamespace(eos_token_id=None)) == [248046]


@pytest.mark.parametrize('token_id', [None, True, -1, '248046'])
def test_invalid_tokenizer_eos_fails_closed(token_id):
    with pytest.raises(ValueError):
        explicit_eos_ids(SimpleNamespace(eos_token_id=token_id),
                         SimpleNamespace(eos_token_id=248044))

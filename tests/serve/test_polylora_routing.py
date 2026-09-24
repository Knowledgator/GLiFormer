import pytest

from gliformer.gliformer import BaseGLiFormer


def test_adapter_ids_follow_filtering_and_minibatches():
    kwargs = BaseGLiFormer._select_valid_forward_kwargs(
        {"adapter_ids": ["a", "drop", "b"]},
        valid_to_orig_idx=[0, 2],
        num_original=3,
    )

    assert kwargs["adapter_ids"] == ["a", "b"]
    assert BaseGLiFormer._batch_forward_kwargs(kwargs, 0, 1, 2)["adapter_ids"] == ["a"]
    assert BaseGLiFormer._batch_forward_kwargs(kwargs, 1, 1, 2)["adapter_ids"] == ["b"]


def test_shared_adapter_is_expanded():
    kwargs = BaseGLiFormer._select_valid_forward_kwargs(
        {"adapter_ids": "legal"},
        valid_to_orig_idx=[1, 2],
        num_original=3,
    )
    assert kwargs["adapter_ids"] == ["legal", "legal"]


def test_adapter_count_is_validated():
    with pytest.raises(ValueError, match="one id per input text"):
        BaseGLiFormer._select_valid_forward_kwargs(
            {"adapter_ids": ["a"]},
            valid_to_orig_idx=[0, 1],
            num_original=2,
        )

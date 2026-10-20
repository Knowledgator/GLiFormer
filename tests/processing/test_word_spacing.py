"""Word boundaries for tokenizers that encode the space in front of a word."""

import pickle

from gliner.data_processing import WordsSplitter

from gliformer.processing.word_spacing import (
    SpaceAwareWordsSplitter,
    SpacedWord,
    restore_word_spaces,
    tokenizer_drops_word_spaces,
)


class _GluingTokenizer:
    """Byte-level BPE behaviour: a word only gets a space if it brings one."""

    def __call__(self, words, is_split_into_words=True, add_special_tokens=False):
        return {"input_ids": list(words)}

    def decode(self, input_ids):
        return "".join(input_ids)


class _SpacingTokenizer(_GluingTokenizer):
    def decode(self, input_ids):
        return " ".join(input_ids)


def _split(text):
    return list(SpaceAwareWordsSplitter(WordsSplitter("whitespace"))(text))


def test_splitter_keeps_tokens_and_offsets():
    text = "Mail  anna-marie@x.io, now!"
    assert [(str(token), start, end) for token, start, end in _split(text)] == list(WordsSplitter("whitespace")(text))


def test_words_remember_the_whitespace_before_them():
    words = [word for word, _, _ in _split("Mail  anna@x.io,\nnow!")]
    assert [str(word) for word in words] == ["Mail", "anna", "@", "x", ".", "io", ",", "now", "!"]
    assert [word.space_before for word in words] == [False, True, False, False, False, False, False, True, False]


def test_restored_words_rebuild_the_text():
    text = "Mail anna@x.io, now!"
    words = [word for word, _, _ in _split(text)]
    assert "".join(restore_word_spaces(words)) == text


def test_plain_words_are_assumed_space_separated():
    assert restore_word_spaces(["a", "b", "c"]) == ["a", " b", " c"]


def test_spacing_survives_slicing_and_pickling():
    words = [word for word, _, _ in _split("one two, three")]
    restored = pickle.loads(pickle.dumps(words[1:]))
    assert all(isinstance(word, SpacedWord) for word in restored)
    assert [word.space_before for word in restored] == [True, False, True]


def test_detects_tokenizers_that_glue_words():
    assert tokenizer_drops_word_spaces(_GluingTokenizer())
    assert not tokenizer_drops_word_spaces(_SpacingTokenizer())

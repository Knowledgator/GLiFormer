"""Keep the spaces between words for tokenizers that encode them.

Text reaches the tokenizer as a list of words. WordPiece and SentencePiece
tokenizers put the word boundary back themselves; byte-level BPE tokenizers
(GPT-2, Qwen, ...) do not, and tokenize ``["quick", "fox"]`` as ``quickfox`` --
text their language model has never seen. For those, each word is handed over
with the whitespace that preceded it in the source text.
"""

from typing import Any, Iterator, List, Sequence, Tuple


class SpacedWord(str):
    """A word that remembers whether whitespace preceded it in its source text."""

    space_before: bool = True

    def __new__(cls, value: str, space_before: bool = True):
        word = super().__new__(cls, value)
        word.space_before = space_before
        return word

    def __reduce__(self):
        return (SpacedWord, (str(self), self.space_before))


class SpaceAwareWordsSplitter:
    """Wraps a words splitter so its words carry ``space_before``."""

    def __init__(self, splitter: Any):
        self.splitter = splitter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.splitter, name)

    def __call__(self, text: str) -> Iterator[Tuple[str, int, int]]:
        for token, start, end in self.splitter(text):
            yield SpacedWord(token, start > 0 and text[start - 1].isspace()), start, end


def tokenizer_drops_word_spaces(tokenizer: Any) -> bool:
    """Whether pre-split words come back glued together from this tokenizer."""
    input_ids = tokenizer(["a", "b"], is_split_into_words=True, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(input_ids).strip() == "ab"


def restore_word_spaces(words: Sequence[str]) -> List[str]:
    """Prefix each word with the space that preceded it.

    Words that did not come from a :class:`SpaceAwareWordsSplitter` (pre-tokenized
    rows) are assumed to be space separated.
    """
    return [
        f" {word}" if getattr(word, "space_before", index > 0) else str(word)
        for index, word in enumerate(words)
    ]

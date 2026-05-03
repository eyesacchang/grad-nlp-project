from datasets import load_dataset


def get_opus100_iterator(lang_pair="en-es", split="train", buffer_size=10000, seed=42):
    """Stream EN-ES sentence pairs from OPUS-100.

    Yields (src_text, tgt_text) tuples. Uses streaming to avoid OOM on full corpus.
    """
    src_lang, tgt_lang = lang_pair.split("-")
    dataset = load_dataset("opus100", lang_pair, split=split, streaming=True)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)

    for example in dataset:
        translation = example["translation"]
        yield translation[src_lang], translation[tgt_lang]


def get_opus100_sample(lang_pair="en-es", n=256, split="train", seed=42):
    """Return a fixed list of N sentence pairs, used for alignment score probing."""
    pairs = []
    for src, tgt in get_opus100_iterator(lang_pair=lang_pair, split=split, seed=seed):
        pairs.append((src, tgt))
        if len(pairs) >= n:
            break
    return pairs

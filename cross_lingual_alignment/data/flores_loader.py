from datasets import load_dataset


# facebook/flores uses a deprecated loading script; openlanguagedata/flores_plus
# is the maintained Parquet version with identical language codes and splits.
FLORES_DATASET = "openlanguagedata/flores_plus"

FLORES_LANG_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "cmn_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "sw": "swh_Latn",
}


def get_flores_pairs(src_lang="en", tgt_lang="es", split="devtest"):
    """Return aligned (src_text, tgt_text) pairs from FLORES-200.

    FLORES-200 is parallel by construction: index i in src matches index i in tgt.
    Use 'devtest' (1012 pairs) for final eval, 'dev' for intermediate checkpoints.
    """
    src_code = FLORES_LANG_CODES[src_lang]
    tgt_code = FLORES_LANG_CODES[tgt_lang]

    src_data = load_dataset(FLORES_DATASET, src_code, split=split)
    tgt_data = load_dataset(FLORES_DATASET, tgt_code, split=split)

    assert len(src_data) == len(tgt_data), "FLORES-200 splits are not the same length"

    pairs = [(src["text"], tgt["text"]) for src, tgt in zip(src_data, tgt_data)]
    return pairs


def get_flores_sentences(lang="en", split="devtest"):
    """Return a list of sentences for a single language."""
    lang_code = FLORES_LANG_CODES[lang]
    data = load_dataset(FLORES_DATASET, lang_code, split=split)
    return [ex["text"] for ex in data]

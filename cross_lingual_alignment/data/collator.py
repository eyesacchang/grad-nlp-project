from transformers import XLMRobertaTokenizerFast


class PairCollator:
    """Tokenizes (src, tgt) sentence pairs for contrastive training."""

    def __init__(self, model_name="xlm-roberta-base", max_length=128):
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, pairs):
        """
        Args:
            pairs: list of (src_text, tgt_text) tuples

        Returns:
            dict with input_ids_src, attention_mask_src, input_ids_tgt, attention_mask_tgt
        """
        src_texts = [p[0] for p in pairs]
        tgt_texts = [p[1] for p in pairs]

        src_enc = self.tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids_src": src_enc["input_ids"],
            "attention_mask_src": src_enc["attention_mask"],
            "input_ids_tgt": tgt_enc["input_ids"],
            "attention_mask_tgt": tgt_enc["attention_mask"],
        }

from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification


# fmt: off
tokenizer = LayoutLMTokenizer.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)
# fmt: on
model = LayoutLMForSequenceClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)

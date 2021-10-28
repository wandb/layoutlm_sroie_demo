from transformers import (
    LayoutLMTokenizer,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
)
from constants import config


tokenizer = LayoutLMTokenizer.from_pretrained(
    config["tokenizer_name"],
)
model = LayoutLMForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=config["model_name"],
    config=LayoutLMConfig(num_labels=config["num_labels"]),
)

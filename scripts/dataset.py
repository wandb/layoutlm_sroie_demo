import json
from pathlib import Path
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer
from transforms import GetTokenBoxesLabels


class SROIE(Dataset):
    def __init__(
        self,
        data_path,
        layoutlm_config_name,
        label_map,
    ):
        super(Dataset, self).__init__()
        self.data_path = Path(data_path)
        tokenizer = LayoutLMTokenizer.from_pretrained(
            layoutlm_config_name,
        )
        self.transform = GetTokenBoxesLabels(
            tokenizer=tokenizer,
            label_map=label_map,
        )
        self.label_map = label_map

        with Path("label_map.json").open("r") as f:
            self.label_map = json.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        local_data_path = self.data_path / f"{metadata['relpath']}"
        if local_data_path.is_file():
            with local_data_path.open("r") as f:
                data = json.load(fp=f)

        output = self.transform(
            data=data,
        )
        # fmt: off
        return {
            k: v
            for k, v
            in output.items()
            if k not in ["token_list", "token_word_map"]
        }
        # fmt: on

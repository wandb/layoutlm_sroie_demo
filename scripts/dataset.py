import json
from pathlib import Path
from torch.utils.data import Dataset


class SROIE(Dataset):
    def __init__(
        self,
        data_path,
        transform,
    ):
        self.data_path = Path(data_path)
        self.filenames_list = [fp.name for fp in self.data_path.glob("*.json")]
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        filename = self.filenames_list[idx]
        local_data_path = self.data_path / filename
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

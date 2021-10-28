import json
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config,
        model,
        dataloader_train,
        dataloader_test,
    ):
        self.config = config
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

    def train(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config["momentum"],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.train()
        self.model.to(device)
        # n_train = len(self.dataloader_train)
        losses = []
        for epoch in range(0, self.config["epochs"]):
            losses.append(0)
            print(f"epoch {epoch}")
            for batch in tqdm(self.dataloader_train):
                batch_input_ids = batch["input_ids"].to(device)
                batch_attention_mask = batch["attention_mask"].to(device)
                batch_token_type_ids = batch["token_type_ids"].to(device)
                batch_bbox = batch["bbox"].to(device)
                batch_labels = batch["labels"].to(device)

                outputs = self.model(
                    input_ids=batch_input_ids,
                    bbox=batch_bbox,
                    attention_mask=batch_attention_mask,
                    token_type_ids=batch_token_type_ids,
                    labels=batch_labels,
                )

                optimizer.zero_grad()
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                # fmt: off
                losses[epoch] += loss.data.cpu().numpy().reshape(1)[0].item()
                # fmt: on

            losses_file = Path("losses.json")
            with losses_file.open("w") as f:
                json.dump(
                    obj=losses,
                    fp=f,
                )

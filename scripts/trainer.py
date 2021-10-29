from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb


class Trainer:
    def __init__(
        self,
        config,
        model,
        dataloader_train,
        dataloader_test,
        run,
    ):
        self.config = config
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.run = run

    def train(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
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
                losses[epoch] += loss.data.cpu().numpy().reshape(1)[0].item()

            wandb.log({"loss": losses[epoch]})

            if (epoch + 1) % self.config["log_freq"] == 0:
                model_checkpoint_artifact = wandb.Artifact(
                    name="LayoutLM",
                    description="checkpoint of LayoutLM trained on SROIE",
                    type="model",
                )
                model_dir = Path.cwd().parent / self.config["model_path"]
                torch.save(
                    self.model.state_dict(),
                    model_dir,
                )
                model_checkpoint_artifact.add_dir(str(model_dir))
                self.run.log_artifact(model_checkpoint_artifact)

            # TODO: add test function and loop

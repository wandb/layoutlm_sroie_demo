from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report


class Trainer:
    def __init__(
        self,
        config,
        model,
        dataloader_train,
        dataloader_test,
        label_encoder,
        run,
    ):
        self.config = config
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.label_encoder = label_encoder
        self.run = run

    def test(self, epoch):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for batch in tqdm(self.dataloader_test):
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

            y_true = batch_labels.squeeze().detach().cpu().numpy().tolist()
            # fmt: off
            y_pred = (
                torch.argmax(outputs.logits.squeeze().detach(), dim=1)
                .cpu()
                .numpy()
                .tolist()
            )
            # fmt: on

            clfn_report = classification_report(
                y_true=y_true,
                y_pred=y_pred,
                output_dict=True,
                labels=list(self.label_encoder.values()),
                target_names=list(self.label_encoder.keys()),
            )
            for label_name in list(self.label_encoder.keys()):
                metrics = clfn_report[label_name]
                self.run.log(
                    {
                        f"{label_name}/precision": metrics["precision"],
                        f"{label_name}/recall": metrics["recall"],
                        f"{label_name}/f1-score": metrics["f1-score"],
                    },
                    step=epoch,
                )

    def train(self):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["learning_rate"],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.train()
        self.model.to(device)
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

            self.run.log(
                {"loss": losses[epoch]},
                step=epoch,
            )

            if (epoch + 1) % self.config["log_freq"] == 0:
                self.test(epoch=epoch)
                # model_checkpoint_artifact = wandb.Artifact(
                #     name="LayoutLM",
                #     description="checkpoint of LayoutLM trained on SROIE",
                #     type="model",
                # )
                model_dir = Path.cwd() / self.config["model_path"]
                model_dir.mkdir(exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    model_dir / "state_dict.pt",
                )
                # model_checkpoint_artifact.add_dir(str(model_dir))
                # self.run.log_artifact(model_checkpoint_artifact)

            if (epoch + 1) == self.config["epochs"]:
                # self.test(epoch=epoch)
                model_checkpoint_artifact = wandb.Artifact(
                    name="LayoutLM",
                    description="checkpoint of LayoutLM trained on SROIE",
                    type="model",
                )
                model_dir = Path.cwd() / self.config["model_path"]
                model_dir.mkdir(exist_ok=True)
                self.model.to("cpu")
                torch.save(
                    self.model.state_dict(),
                    model_dir / "state_dict.pt",
                )
                model_checkpoint_artifact.add_dir(str(model_dir))
                self.run.log_artifact(model_checkpoint_artifact)

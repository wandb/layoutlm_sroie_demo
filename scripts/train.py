from pathlib import Path
import json
import torch
from torch.utils.data import random_split, DataLoader
import wandb
import tqdm
from dataset import SROIE
from transforms import GetTokenBoxesLabels
from model import (
    tokenizer,
    model,
)
from trainer import Trainer
from constants import config, color_map


def main(
    config,
    run,
):
    with (Path.cwd() / "label_encoder.json").open("r") as f:
        label_encoder = json.load(f)

    transform = GetTokenBoxesLabels(
        tokenizer=tokenizer,
        label_encoder=label_encoder,
    )
    dataset = SROIE(
        data_path=config["data_path"],
        transform=transform,
    )
    dataset_length = len(dataset)
    train_length = int(config["pct_train"] * dataset_length)
    test_length = dataset_length - train_length
    dataset_train, dataset_test = random_split(
        dataset,
        (train_length, test_length),
    )

    dataloader_train = DataLoader(
        # dataset_train,
        dataset,
        sampler=list(range(0, 8)),
        # shuffle=True,
        batch_size=config["batch_size_train"],
        drop_last=False,
    )

    dataloader_test = DataLoader(
        # dataset_test,
        dataset,
        sampler=list(range(0, 8)),
        # shuffle=True,
        batch_size=config["batch_size_test"],
        drop_last=False,
    )

    trainer = Trainer(
        config=config,
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        label_encoder=label_encoder,
        run=run,
    )

    trainer.train()

    for batch in tqdm(dataloader_test, "final"):
        batch_input_ids = batch["input_ids"]
        batch_attention_mask = batch["attention_mask"]
        batch_token_type_ids = batch["token_type_ids"]
        batch_bbox = batch["bbox"]
        batch_labels = batch["labels"]

        outputs = model(
            input_ids=batch_input_ids,
            bbox=batch_bbox,
            attention_mask=batch_attention_mask,
            token_type_ids=batch_token_type_ids,
            labels=batch_labels,
        )

        # y_true = batch_labels.squeeze().detach().numpy().tolist()
        # fmt: off
        y_pred = (
            torch.argmax(outputs.logits.squeeze().detach(), dim=1)
            .numpy()
            .tolist()
        )
        # fmt: on

        bbox_list = batch_bbox.detach().numpy().tolist()
        selected_boxes = [
            {"bbox": bbox_list[index], "pred": pred, "color": color_map[pred]}
            for index, pred in enumerate(
                [y for y in y_pred if y != label_encoder["none"]]
            )
        ]
        print(selected_boxes)


if __name__ == "__main__":
    run = wandb.init(
        project="layoutlm_sroie_demo",
        entity="wandb-data-science",
        job_type="train",
    )
    wandb.config = config
    main(
        config=config,
        run=run,
    )

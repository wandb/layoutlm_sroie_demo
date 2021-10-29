from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import wandb
from tqdm import tqdm
import cv2
from PIL import Image
from dataset import SROIE
from transforms import GetTokenBoxesLabels
from model import (
    tokenizer,
    model,
)
from trainer import Trainer
from constants import config, color_map, task_1_dir


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
        config=config,
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
        sampler=list(range(0, 4)),
        # shuffle=True,
        batch_size=config["batch_size_train"],
        drop_last=False,
    )

    dataloader_test = DataLoader(
        # dataset_test,
        dataset,
        sampler=list(range(0, 4)),
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

    annotated_images = []
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

        # fmt: off
        y_pred = (
            torch.argmax(outputs.logits.squeeze().detach(), dim=1)
            .numpy()
            .tolist()
        )
        # fmt: on

        bbox_list = batch_bbox.squeeze().detach().numpy().tolist()
        selected_boxes = [
            {"bbox": bbox_list[index], "pred": pred, "color": color_map[pred]}
            for index, pred in enumerate(
                [y for y in y_pred if y != label_encoder["none"]]
            )
        ]
        image_width = batch["image_width"]
        image_height = batch["image_height"]
        # fmt: off
        image = Image.open(
            Path.cwd().parent
            / "data_raw"
            / task_1_dir
            / f"{batch['id'][0]}.jpg"
        )
        # fmt: on
        image_arr = np.asarray(image)
        # convert rgb array to opencv's bgr format
        image_arr_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
        for box in selected_boxes:
            bbox = box["bbox"]
            color = box["color"]
            # fmt: off
            cv2.rectangle(
                image_arr_bgr,
                (
                    image_width * (bbox[0] / 1000.0),
                    image_height * (bbox[1] / 1000.0)
                ),
                (
                    image_width * (bbox[2] / 1000.0),
                    image_height * (bbox[3] / 1000.0)
                ),
                color=color,
                thickness=3,
            )
            # fmt: on
            image_arr = cv2.cvtColor(image_arr_bgr, cv2.COLOR_BGR2RGB)

        # convert back to Image object
        image = Image.fromarray(image_arr)
        annotated_images.append(
            {
                "id": batch["id"],
                "image_annotated": image,
            }
        )

    table = wandb.Table(
        columns=["id", "image_annotated"],
        data=[
            [
                sample["id"],
                wandb.Image(sample["image_annotated"]),
            ]
            for sample in annotated_images
        ],
    )
    run.log({"LayoutLM on SROIE: Annotations": table})


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

from pathlib import Path
import json
import torch
from torch.utils.data import random_split, DataLoader
import wandb
from tqdm import tqdm
from PIL import Image
import cloudpickle
import objects
from objects.dataset import SROIE
from objects.transforms import GetTokenBoxesLabels
from objects.model import (
    tokenizer,
    model,
)
from objects.trainer import Trainer
from objects.constants import config, task_1_dir


def main(
    config,
    run,
):
    with (Path.cwd() / "scripts" / "label_encoder.json").open("r") as f:
        label_encoder = json.load(f)

    transform = GetTokenBoxesLabels(
        tokenizer=tokenizer,
        label_encoder=label_encoder,
    )
    transform_dir = Path.cwd() / "transforms"
    transform_dir.mkdir(exist_ok=True)
    cloudpickle.register_pickle_by_value(objects)
    with (transform_dir / "transform.cloudpickle").open("wb") as f:
        cloudpickle.dump(
            obj=transform,
            file=f,
        )
    cloudpickle.unregister_pickle_by_value(objects)

    transform_artifact = wandb.Artifact(
        name="GetTokenBoxesLabels",
        description="transform of GetTokenBoxesLabels",
        type="callable class",
    )
    transform_artifact.add_dir(str(transform_dir))
    run.log_artifact(transform_artifact)

    dataset = SROIE(
        run=run,
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
        dataset_train,
        shuffle=True,
        batch_size=config["batch_size_train"],
        drop_last=False,
    )

    dataloader_test = DataLoader(
        dataset_test,
        shuffle=True,
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

    label_encoder_inv = {value: key for key, value in label_encoder.items()}

    sroie = Path.cwd() / config["data_raw_path"]
    sroie.mkdir(exist_ok=True)
    artifact_data_raw = run.use_artifact(
        f"{run.entity}/{run.project}/data_raw:latest",
        type="dataset",
    )
    artifact_data_raw.download(
        root=str(Path.cwd()),
    )

    class_set = wandb.Classes(
        [
            {
                "name": key,
                "id": value,
            }
            for key, value in label_encoder.items()
        ]
    )

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
            {
                "bbox": bbox_list[index],
                "pred": pred,
            }
            for index, pred in [
                (i, y_pred[i])
                for i in range(len(y_pred))
                if y_pred[i] != label_encoder["none"]
            ]
        ]
        image_width = batch["image_width"]
        image_height = batch["image_height"]
        # fmt: off
        image = Image.open(
            Path.cwd()
            / config["data_raw_path"]
            / task_1_dir
            / f"{batch['id'][0]}.jpg"
        )
        # fmt: on

        annotated_images.append(
            {
                "id": batch["id"],
                "image": image,
                "annotations": {
                    "predictions": {
                        "box_data": [
                            {
                                "position": {
                                    # fmt: off
                                    "minX": int(
                                        image_width * (
                                            box["bbox"][0] / 1000.0
                                        )
                                    ),
                                    "maxX": int(
                                        image_width * (
                                            box["bbox"][2] / 1000.0
                                        )
                                    ),
                                    "minY": int(
                                        image_height * (
                                            box["bbox"][1] / 1000.0
                                        )
                                    ),
                                    "maxY": int(
                                        image_height * (
                                            box["bbox"][3] / 1000.0
                                        )
                                    ),
                                    # fmt: on
                                },
                                "domain": "pixel",
                                "class_id": box["pred"],
                                "box_caption": label_encoder_inv[box["pred"]],
                            }
                            for box in selected_boxes
                        ],
                        "class_labels": label_encoder_inv,
                    }
                },
            }
        )

    table = wandb.Table(
        columns=["id", "image_annotated"],
        data=[
            [
                sample["id"],
                wandb.Image(
                    data_or_path=sample["image"],
                    boxes=sample["annotations"],
                    classes=class_set,
                ),
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

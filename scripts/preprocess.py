from pathlib import Path
import json
import re
from PIL import Image
import wandb


def parse_line(line, regex=r"((.*?,){8})"):
    matches = re.search(regex, line)

    if matches:
        end = matches.end()
        match = matches.group()
        match = match.strip(",")
        coords = match.split(",")

        return {
            # top left corner coords
            "x1": coords[0],
            "y1": coords[1],
            # bottom right corner coords
            "x2": coords[4],
            "y2": coords[5],
            "text": line[end:],
        }
    else:
        return None


def parse_ocr(txt: str) -> list:
    lines = txt.split("\n")
    ocr = []
    for line in lines:
        content = line.split(",")
        if len(content) >= 8:
            ocr.append(
                parse_line(
                    line=line,
                )
            )

    return ocr


def split_text_and_bbox(record: dict) -> list:
    words = record["text"].split(" ")
    n_words = len(words)
    x1 = int(record["x1"])
    y1 = int(record["y1"])
    x2 = int(record["x2"])
    y2 = int(record["y2"])
    step = int((x2 - x1) / float(n_words))
    return [
        {
            "x1": x1 + step * index,
            "y1": y1,
            "x2": x1 + step * (index + 1),
            "y2": y2,
            "text": words[index],
        }
        for index in range(n_words)
    ]


def main(run):
    sroie = Path.cwd().parent / "data_raw"
    task_1 = sroie / "0325updated.task1train(626p)"
    task_2 = sroie / "0325updated.task2train(626p)"

    artifact_data_raw = wandb.Artifact(
        name="data_raw",
        type="dataset",
    )
    artifact_data_raw.add_dir(
        local_path=str((Path.cwd().parent / "data_raw")),
        name="SROIE",
    )
    run.log_artifact(artifact_data_raw)

    data = []
    label_set = {"none"}

    jpgs = task_1.glob("**/*.jpg")
    for jpg in jpgs:
        if "(" not in jpg.name:
            image = Image.open(jpg)
            width, height = image.size

            ocr = parse_ocr((task_1 / f"{jpg.stem}.txt").read_text())
            labels = json.loads((task_2 / f"{jpg.stem}.txt").read_text())
            token_data = []
            for key in labels.keys():
                label_set.add(key)

            for element in ocr:
                tokens = split_text_and_bbox(element)
                for token in tokens:
                    token["label_name"] = "none"
                    for label_name, label_value in labels.items():
                        if label_value in element["text"]:
                            token["label_name"] = label_name

                    token_data.append(token)

            data.append(
                {
                    "id": jpg.stem,
                    "image_width": width,
                    "image_height": height,
                    "data": token_data,
                }
            )

    # fmt: off
    label_encoder = {
        label_name: label
        for label, label_name in enumerate(label_set)
    }
    # fmt: on
    with (Path.cwd() / "label_encoder.json").open("w") as f:
        json.dump(
            obj=label_encoder,
            fp=f,
        )

    for sample in data:
        for element in sample["data"]:
            element["label"] = label_encoder.get(
                element["label_name"],
            )

    output_path = Path.cwd().parent / "data"

    for sample in data:
        file_id = sample["id"]
        filepath = output_path / f"{file_id}.json"
        with filepath.open("w") as f:
            json.dump(obj=sample, fp=f)

    artifact_data = wandb.Artifact(
        name="data",
        type="dataset",
    )
    artifact_data.add_dir(
        local_path=str(Path.cwd().parent / "data"),
        name="SROIE_LayoutLM",
    )
    run.log_artifact(artifact_data)

    table = wandb.Table(
        columns=["id", "image", "height", "width"],
        data=[
            [
                sample["id"],
                wandb.Image(str(task_1 / f"{sample['id']}.jpg")),
                sample["image_height"],
                sample["image_height"],
            ]
            for sample in data
        ],
    )
    run.log({"LayoutLM on SROIE": table})


if __name__ == "__main__":
    run = wandb.init(
        project="layoutlm_sroie_demo",
        entity="wandb-data-science",
        job_type="preprocess",
    )
    main(run=run)

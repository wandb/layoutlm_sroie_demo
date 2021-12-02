# LayoutLM on SROIE

This code fine-tunes [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) on the [SROIE](https://rrc.cvc.uab.es/?ch=13) scanned receipts data, and uses [Weights & Biases](https://wandb.ai/site) to log losses and metrics during training, and annotated images with bounding box predictions. Here is the accompanying [Report](https://wandb.ai/wandb-data-science/layoutlm_sroie_demo/reports/Fine-tuning-LayoutLM-on-SROIE-Information-Extraction-from-Scanned-Receipts--VmlldzoxMjI5NzE2).

### **Example annotated receipt**
<p align="center">
<img src="images/example_receipt_w_bboxes.png" height=400></img>
</p>


### **Plots of training metrics**
<p align="center">
<img src="iamges/../images/metrics_layoutlm_wandb.png" width=500></img>
</p>

## Getting started

First, make sure to install the [pipenv](https://github.com/pypa/pipenv) environment, using `pipenv install`. This requires pipenv to have access to python 3.9. To install and manage different python versions, try out [pyenv](https://github.com/pyenv/pyenv). All instructions below assume the `pipenv` environment is activated; to activate, run `pipenv shell`.

### Preprocessing

The preprocessing for this slightly nonstandard, since the OCR and labels are given in a format that is not consistent with the per-token level classification setup that LayoutLM requires. More details given in this [section](https://wandb.ai/wandb-data-science/layoutlm_sroie_demo/reports/Fine-tuning-LayoutLM-on-SROIE-Information-Extraction-from-Scanned-Receipts--VmlldzoxMjI5NzE2#general-pipeline) of the report.

To run the preprocessing step, from the base directory, run

```python
python -m scripts.preprocess
```

### Training

To train, run the following command from base directory

```python
python -m scripts.train
```

### Objects

The different objects used in preprocessing the data and training the model are contained in the `objects` directory. Below is a rough listing of the files and objects contained

- objects
  - constants.py
    - config
    - task_1_dir
  - dataset.py
    - SROIE(Dataset)
  - model.py
    - tokenizer
    - model
  - trainer.py
    - Trainer
  - transforms.py
    - GetTokenBoxesLabels


#### **GetTokenBoxesLabels**

Special attention should be brought to the callable class `GetTokenBoxesLabels` defined in `transforms.py`. This does three main things

1. Tracks tokenization of words and appropriately duplicates the bounding boxes accommodate the tokenized sequence.
2. Pads the input sequence to the max length allowable by the tokenizer (here it is BERTTokenizer, so 256).
3. Normalizes coordinates to be between 0 and 1000. This is required by LayoutLM.

An example of why #1 is necessary might be if the sequence of `(word, bbox)` pairs corresponding to a segment of text on a document is

```python
[("I", [100, 100, 120, 150]), ("am", [130, 100, 160, 150]), ("sleeping", [140, 100, 280, 150])]
```

Here the bounding box coordinates are in the format `[x1, y1, x2, y2]`, where `x1` and `x2` are the left- and right- most coordinates of the bounding box; and similarly `y1` and `y2` are the top- and bottom- most coordinates. The tokenizer itself operates only on the sequence of words

```I am sleeping```

and returns the sequence of tokens

```I am sleep ##ing```

But does not operate on the bounding boxes. For the purposes of LayoutLM, we want the `(token, bbox)` sequence to be

```python
[("I", [100, 100, 120, 150]), ("am", [130, 100, 160, 150]), ("sleep", [140, 100, 280, 150]), ("##ing", [140, 100, 280, 150])]
```

`GetTokenBoxesLabels` takes care of this.

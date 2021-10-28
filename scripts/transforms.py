import torch


class GetTokenBoxesLabels:
    """Retreives the bounding boxes and labels for tokens coming
    from OCR. Appropriate to be used with LayoutLM.
    """

    def __init__(
        self,
        tokenizer,
        # label_map: dict,
    ) -> None:
        self.tokenizer = tokenizer
        # self.label_map = label_map
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_box = [0, 0, 0, 0]
        self.sep_token_box = [1000, 1000, 1000, 1000]
        self.pad_token_label = 0
        self.pad_token = 0
        self.pad_token_type_id = 0
        self.pad_token_box = [0, 0, 0, 0]
        self.special_tokens_count = 2
        self.max_seq_length = self.tokenizer.model_max_length

    def transform(
        self,
        data: dict,
    ) -> dict:
        """Retereives the bounding boxes and labels for tokens coming
        from OCR.

        Args:
            data (dict): dictionary containing data from OCR.

        Returns:
            dict: dictionary containing tokenized text with appropriate
            boudning boxes and labels associated.
        """
        total_token_length = 0
        image_width = float(data["image_width"])
        image_height = float(data["image_height"])
        words = []
        tokens = []
        token_boxes = []
        token_labels = []
        token_word_map = [1]  # runtime encoding of token->word mappings
        for elt in data["data"]:
            word = elt["text"]
            word_tokens = self.tokenizer.tokenize(word)
            num_word_tokens = len(word_tokens)
            total_token_length += num_word_tokens
            # max_seq_length minus one for padding
            if total_token_length < self.max_seq_length - 1:
                words.append(word)
                tokens.extend(word_tokens)
                # label = self.label_map.get(elt.get("label", "0"), 0)
                label = elt.get("label")
                # fmt: off
                nrmlzd_word_box = [
                    int(1000 * (float(elt["left"]) / image_width)),
                    int(1000 * (float(elt["top"]) / image_height)),
                    int(
                        1000 * (
                            (float(elt["left"]) + float(elt["width"]))
                            /
                            image_width
                        )
                    ),
                    int(
                        1000
                        * (
                            (float(elt["top"]) + float(elt["height"]))
                            /
                            image_height
                        )
                    ),
                ]
                # fmt: on

                token_boxes.extend([nrmlzd_word_box] * num_word_tokens)
                token_labels.extend([label] * num_word_tokens)
                token_word_map.append(num_word_tokens)

        encoding = self.tokenizer(
            " ".join(words),
            add_special_tokens=True,
            max_length=self.max_seq_length,
            truncation=True,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]

        # add special token boxes and labels
        token_boxes = [self.cls_token_box, *token_boxes, self.sep_token_box]
        token_labels = [
            self.pad_token_label,
            *token_labels,
            self.pad_token_label,
        ]

        # pad sequences
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.pad_token] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [self.pad_token_type_id] * padding_length
        token_labels += [self.pad_token_label] * padding_length
        token_boxes += [self.pad_token_box] * padding_length

        input_ids = torch.tensor(input_ids).type(torch.LongTensor)
        token_labels = torch.tensor(token_labels).type(torch.LongTensor)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)
        bbox = torch.tensor(token_boxes)

        return {
            "image_width": image_width,
            "image_height": image_height,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "token_list": tokens,
            "token_word_map": token_word_map,
            "bbox": bbox,
            "labels": token_labels,
        }

import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_len=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.abstract = dataframe.abstract
        self.labels = self.data.label
        self.max_len = max_len

    def labels(self):
        return self.labels

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, idx):
        abstract = str(self.abstract[idx])
        abstract = " ".join(abstract.split())

        inputs = self.tokenizer.encode_plus(
            abstract,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

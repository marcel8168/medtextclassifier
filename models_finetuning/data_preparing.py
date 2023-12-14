from models_finetuning.dataset import Dataset
from torch.utils.data import DataLoader


def get_dataloader(texts, targets, tokenizer, batch_size, max_len, num_workers=0):
    dataset = Dataset(texts.to_numpy(), targets, tokenizer, max_len)
    params = {
        "batch_size": batch_size,
        "num_workers": num_workers
    }
    dataloader = DataLoader(dataset, **params)

    return dataloader

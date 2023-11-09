from mteb import MTEB
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class MyModel():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("llmrails/ember-v1")
        self.model = AutoModel.from_pretrained("llmrails/ember-v1")

    def average_pool(self, last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict['attention_mask'])

        return embeddings


model = MyModel()
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model)

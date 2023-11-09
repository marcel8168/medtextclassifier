from mteb import MTEB
import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from collections import OrderedDict


class MyModel():
    def __init__(self) -> None:
        self.model = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True,)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def bert_text_preparation(self, text, tokenizer):
        """
        Preprocesses text input in a way that BERT can interpret.
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)

        # convert inputs to tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensor

    def get_bert_embeddings(self, tokens_tensor, segments_tensor, model):
        """
        Obtains BERT embeddings for tokens.
        """
        # gradient calculation id disabled
        with torch.no_grad():
            # obtain hidden states
            outputs = model(tokens_tensor, segments_tensor)
            hidden_states = outputs[2]
        # concatenate the tensors for all layers
        # use "stack" to create new dimension in tensor
        token_embeddings = torch.stack(hidden_states, dim=0)
        # remove dimension 1, the "batches"
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # swap dimensions 0 and 1 so we can loop over tokens
        token_embeddings = token_embeddings.permute(1, 0, 2)
        # intialized list to store embeddings
        token_vecs_sum = []
        # "token_embeddings" is a [Y x 12 x 768] tensor
        # where Y is the number of tokens in the sentence
        # loop over tokens in sentence
        for token in token_embeddings:
            # "token" is a [12 x 768] tensor
            # sum the vectors from the last four layers
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        return token_vecs_sum

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        list_token_embeddings = []
        for sentence in sentences:
            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(
                sentence, self.tokenizer)
            list_token_embeddings.append(self.get_bert_embeddings(
                tokens_tensor, segments_tensors, self.model))
    
        return np.array(list_token_embeddings)


model = MyModel()
evaluation = MTEB(tasks=["Banking77Classification"])
evaluation.run(model)

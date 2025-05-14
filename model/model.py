import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

def average_pool(last_hidden_states: Tensor,
                attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class BuzzSearchTunedE5(torch.nn.Module):
    def __init__(self, model_name = "intfloat/multilingual-e5-small"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, x):
        query = x['query']
        passage = x['passage']

        query_embeddings = self.model(**query)
        passage_embeddings = self.model(**passage)

        query_embeddings = average_pool(query_embeddings.last_hidden_state, x['query']['attention_mask'])
        passage_embeddings = average_pool(passage_embeddings.last_hidden_state, x['passage']['attention_mask'])

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

        return query_embeddings, passage_embeddings
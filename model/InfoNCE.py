import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss function for self-supervised learning.
    This implementation uses in-batch negatives, where for each query in a batch,
    its corresponding positive passage is known, and all other passages in the batch
    are treated as negative samples.

    Args:
        temperature (float): Temperature scaling factor. A lower temperature makes
                             the model more sensitive to distinguishing hard negatives.
                             Common values are between 0.01 and 0.1.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        # CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
        # It is useful when training a classification problem with C classes.
        # Here, for each query, we are trying to classify which passage (out of all passages in the batch)
        # is its true positive pair.
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_embeddings: Tensor, passage_embeddings: Tensor) -> Tensor:
        """
        Calculates the InfoNCE loss.

        Args:
            query_embeddings (Tensor): A tensor of query embeddings.
                                       Shape: (batch_size, embedding_dimension).
                                       Assumed to be L2 normalized.
            passage_embeddings (Tensor): A tensor of passage embeddings.
                                        Shape: (batch_size, embedding_dimension).
                                        Assumed to be L2 normalized.
                                        passage_embeddings[i] is the positive pair for query_embeddings[i].

        Returns:
            Tensor: The calculated InfoNCE loss (a scalar).
        """
        # Ensure embeddings are on the same device
        if query_embeddings.device != passage_embeddings.device:
            raise ValueError("Query and passage embeddings must be on the same device.")

        batch_size = query_embeddings.shape[0]

        # Calculate cosine similarity between all query and passage embeddings.
        # Since embeddings are assumed to be L2 normalized, matmul is equivalent to cosine similarity.
        # similarity_matrix[i, j] will be the similarity between query_i and passage_j.
        similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) # Shape: (batch_size, batch_size)

        # Scale logits by temperature
        # This controls the sharpness of the distribution. Lower temperature -> sharper distribution.
        scaled_logits = similarity_matrix / self.temperature

        # Create labels for cross-entropy.
        # For each query_i (row i in scaled_logits), the target (positive passage) is at column i.
        # So, labels are [0, 1, 2, ..., batch_size-1].
        labels = torch.arange(batch_size, device=scaled_logits.device)

        # Calculate cross-entropy loss.
        # The CrossEntropyLoss expects raw logits (it applies softmax internally).
        loss = self.cross_entropy_loss(scaled_logits, labels)

        return loss
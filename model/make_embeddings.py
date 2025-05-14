import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset import FullDataset 
from model import BuzzSearchTunedE5, average_pool
import functools
import json
import numpy as np

DEFAULT_MODEL_SAVE_PATH = "best_colab_e5_model_recall5.pth" # Path to your fine-tuned model
DEFAULT_HTML_PATH = "/Users/siddhantagarwal/Desktop/Programming/mini-exa-clone/data_prep/final_data_prep/data/html_final.jsonl"
DEFAULT_REDDIT_PATH = "/Users/siddhantagarwal/Desktop/Programming/mini-exa-clone/data_prep/final_data_prep/data/reddit_final.jsonl"
DEFAULT_MODEL_NAME = 'intfloat/multilingual-e5-small'

ALL_DOC_EMBEDDINGS_NPY_PATH = "all_document_embeddings.npy"
ALL_DOC_IDS_JSON_PATH = "all_document_ids.json"

BATCH_SIZE_FOR_EMBEDDING_GENERATION = 32


def find_embedding(query, passage):
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    passage_inputs = tokenizer(passage, return_tensors="pt", padding=True, truncation=True)

    query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
    passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}

    model_input = {
        'query': query_inputs,
        'passage': passage_inputs
    }

    with torch.no_grad():
        query_embeddings, passage_embeddings = model(model_input)

    return query_embeddings, passage_embeddings


def passage_embedding_collate_fn(batch, tokenizer_func):
    if tokenizer_func is None:
        raise RuntimeError("Tokenizer function is not provided to collate_fn.")

    batched_passage_texts = []
    batched_ids = []

    for row in batch:
        batched_passage_texts.append(row['passage'])
        batched_ids.append(row['id']) 
    
    tokenized_passages = tokenizer_func(
        batched_passage_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    return {
        "passage": {
            "input_ids": tokenized_passages['input_ids'], 
            "attention_mask": tokenized_passages['attention_mask']
        },
        "ids": batched_ids
    }

def generate_and_save_all_document_embeddings(
    full_dataset: FullDataset, 
    model_instance: BuzzSearchTunedE5, 
    tokenizer_instance: AutoTokenizer,
    device_to_use: torch.device, 
    embeddings_save_path: str, 
    ids_save_path: str,
    batch_size: int
):
    model_instance.eval()
    all_passage_embeddings_list = []
    all_passage_ids_list = []

    collate_fn_with_tokenizer = functools.partial(passage_embedding_collate_fn, tokenizer_func=tokenizer_instance)
    
    document_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=0 
    )

    with torch.no_grad():
        for batch_data in tqdm(document_loader, desc="Generating Document Embeddings"):
            passage_inputs_tokenized = {k: v.to(device_to_use) for k, v in batch_data['passage'].items()}
    
            ids_batch = batch_data['ids'] # Retrieve the IDs for this batch

            model_input = {
                'query': passage_inputs_tokenized,
                'passage': passage_inputs_tokenized
            }

            query_embeddings, passage_embeddings = model_instance(model_input)
            
            all_passage_embeddings_list.append(passage_embeddings.cpu())
            all_passage_ids_list.extend(ids_batch)

    if not all_passage_embeddings_list:
        print("No embeddings were generated. Check dataset and model.")
        return

    final_embeddings_tensor = torch.cat(all_passage_embeddings_list, dim=0)
    final_embeddings_numpy = final_embeddings_tensor.numpy()

    # Save embeddings as .npy file
    print(f"Saving {final_embeddings_numpy.shape[0]} document embeddings to {embeddings_save_path}...")
    np.save(embeddings_save_path, final_embeddings_numpy)
    print(f"Embeddings saved to {embeddings_save_path}")

    # Save corresponding IDs as .json file
    print(f"Saving {len(all_passage_ids_list)} document IDs to {ids_save_path}...")
    with open(ids_save_path, 'w', encoding='utf-8') as f_ids:
        json.dump(all_passage_ids_list, f_ids, indent=2) # Using indent for readability
    print(f"Document IDs saved to {ids_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    state_dict = torch.load(DEFAULT_MODEL_SAVE_PATH, map_location=torch.device('cpu'))
    model = BuzzSearchTunedE5(model_name=DEFAULT_MODEL_NAME)
    model.load_state_dict(state_dict)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)

    dataset = FullDataset(DEFAULT_HTML_PATH, DEFAULT_REDDIT_PATH)

    generate_and_save_all_document_embeddings(dataset, model, tokenizer, device, ALL_DOC_EMBEDDINGS_NPY_PATH, ALL_DOC_IDS_JSON_PATH, BATCH_SIZE_FOR_EMBEDDING_GENERATION)
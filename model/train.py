import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset import FullDataset
from model import BuzzSearchTunedE5
from InfoNCE import InfoNCELoss
import argparse
import functools

DEFAULT_HTML_PATH = "/Users/siddhantagarwal/Desktop/Programming/mini-exa-clone/data_prep/final_data_prep/data/html_final.jsonl"
DEFAULT_REDDIT_PATH = "/Users/siddhantagarwal/Desktop/Programming/mini-exa-clone/data_prep/final_data_prep/data/reddit_final.jsonl"
DEFAULT_MODEL_NAME = 'intfloat/multilingual-e5-small'
DEFAULT_MODEL_SAVE_PATH = "best_BuzzSearch_e5_model.pth"

def custom_collate_fn(batch, tokenizer):
    batched_query = []
    batched_passage = []
    for row in batch:
        batched_query.append(row['query'])
        batched_passage.append(row['passage'])
    
    tokenized_query = tokenizer(batched_query, max_length=512, padding=True, truncation=True, return_tensors='pt')
    tokenized_passage = tokenizer(batched_passage, max_length=512, padding=True, truncation=True, return_tensors='pt')

    return {
        "query": {
            "input_ids":tokenized_query['input_ids'], 
            "attention_mask":tokenized_query['attention_mask']
        },
        "passage": {
            "input_ids":tokenized_passage['input_ids'], 
            "attention_mask":tokenized_passage['attention_mask']
        }
    }
    
def train_epoch(model, data_loader, criterion, optimizer, device, epoch_num, num_epochs):
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num}/{num_epochs} [Training]", leave=False)

    for batch_idx, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()

        query_inputs = {k: v.to(device) for k, v in batch['query'].items()}
        passage_inputs = {k: v.to(device) for k, v in batch['passage'].items()}

        model_input = {
            'query': query_inputs,
            'passage': passage_inputs
        }
        
        query_embeddings, passage_embeddings = model(model_input)
        loss = criterion(query_embeddings, passage_embeddings)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        train_progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(data_loader)
    return avg_train_loss

def evaluate_epoch(model, data_loader, criterion, device, epoch_num, num_epochs):
    model.eval()
    total_val_recall_hits = 0
    total_val_queries = 0
    val_progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num}/{num_epochs} [Validation]", leave=False)

    with torch.no_grad():
        for batch in val_progress_bar:
            query_inputs = {k: v.to(device) for k, v in batch['query'].items()}
            passage_inputs = {k: v.to(device) for k, v in batch['passage'].items()}
            
            model_input = {
                'query': query_inputs,
                'passage': passage_inputs
            }

            query_embeddings, passage_embeddings = model(model_input)

            similarities = torch.matmul(query_embeddings, passage_embeddings.T)
            _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
            labels = torch.arange(query_embeddings.size(0), device=device)

            for i in range(query_embeddings.size(0)):
                true_positive_idx = labels[i]
                if true_positive_idx in sorted_indices[i, :10]:
                    total_val_recall_hits += 1
            total_val_queries += query_embeddings.size(0)
            
    avg_val_recall_at_10 = (total_val_recall_hits / total_val_queries) if total_val_queries > 0 else 0.0
    
    return avg_val_recall_at_10

def main(args):
    global tokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    generator = torch.Generator().manual_seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = FullDataset(args.html_path, args.reddit_path)

    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3], generator=generator)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    collate_fn_with_tokenizer = functools.partial(custom_collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn= collate_fn_with_tokenizer, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn= collate_fn_with_tokenizer, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    model = BuzzSearchTunedE5(model_name=args.model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = InfoNCELoss(temperature=args.temperature)

    best_val_recall_at_10 = 0.0
    print("Starting training...")

    for epoch in range(1, args.num_epochs + 1):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.num_epochs)
        print(f"\nEpoch {epoch}/{args.num_epochs} - Training Loss: {avg_train_loss:.4f}")

        avg_val_recall_at_10 = evaluate_epoch(model, val_loader, criterion, device, epoch, args.num_epochs)
        print(f"Epoch {epoch}/{args.num_epochs} - Validation Recall@10: {avg_val_recall_at_10:.4f}")

        if avg_val_recall_at_10 > best_val_recall_at_10:
            best_val_recall_at_10 = avg_val_recall_at_10
            torch.save(model.state_dict(), args.model_save_path)
            print(f"New best model saved to {args.model_save_path} with Recall@10: {best_val_recall_at_10:.4f}")

    print("\nTraining complete.")
    print(f"Best Validation Recall@10: {best_val_recall_at_10:.4f}")
    print(f"Best model saved at: {args.model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BuzzSearch E5 model for link prediction.")
    
    parser.add_argument("--html_path", type=str, default=DEFAULT_HTML_PATH, help="Path to the HTML processed JSONL data.")
    parser.add_argument("--reddit_path", type=str, default=DEFAULT_REDDIT_PATH, help="Path to the Reddit processed JSONL data.")
    
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the pre-trained model from Hugging Face.")
    
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for InfoNCE loss.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train_split_ratio", type=float, default=0.7, help="Proportion of data for training (0.0 to 1.0).")
    
    parser.add_argument("--model_save_path", type=str, default=DEFAULT_MODEL_SAVE_PATH, help="Path to save the best model.")
    
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for DataLoader. Set to 0 for main process only.")

    args = parser.parse_args()
    
    main(args)
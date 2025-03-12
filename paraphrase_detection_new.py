'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
import time

TQDM_DISABLE = True
start_time = time.time()

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    last_hidden_state = self.gpt(input_ids, attention_mask)['last_token'] # [B, H]
    return self.gpt.hidden_state_to_token(last_hidden_state)


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn, num_workers=4)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn, num_workers=4)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)
  a1, a2 = args.a1, args.a2

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01) # increase weight_decay for smaller datasets
  best_dev_acc = 0
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # NOTE: our customization - training regularization via contrastive loss
      labels_float = (labels == 8505).float() # 1 if paraphrase, 0 otherwise
      s1_ids, s1_mask, s2_ids, s2_mask = batch['token_ids_s1'], batch['attention_mask_s1'], batch['token_ids_s2'], batch['attention_mask_s2']
      s1_ids = s1_ids.to(device)
      s1_mask = s1_mask.to(device)
      s2_ids = s2_ids.to(device)
      s2_mask = s2_mask.to(device)

      s1_hidden_state = model.gpt(s1_ids, s1_mask)['last_token'] # [B, H]
      s2_hidden_state = model.gpt(s2_ids, s2_mask)['last_token'] # [B, H]
      
      # Check for NaN in hidden states
      if torch.isnan(s1_hidden_state).any() or torch.isnan(s2_hidden_state).any():
          nan_indices_s1 = torch.where(torch.isnan(s1_hidden_state).any(dim=1))[0]
          nan_indices_s2 = torch.where(torch.isnan(s2_hidden_state).any(dim=1))[0]
          nan_indices = torch.unique(torch.cat([nan_indices_s1, nan_indices_s2]))
          sentence_ids = batch['sent_ids'][nan_indices]
          
          # Debug the debugging: Which samples have NaNs?
          nan_mask_s1 = torch.isnan(s1_hidden_state).any(dim=1)
          print(f"s1_hidden_state shape: {s1_hidden_state.shape}, NaN mask shape: {nan_mask_s1.shape}, count: {nan_mask_s1.sum().item()}")
          
          print(f"NaN detected in hidden states, epoch: {epoch}, batch: {num_batches}, problematic sentence_ids:\n{sentence_ids}")
          continue # Skip this batch
      
      # compute the cosine similarity between the two hidden states
      cos_sim = cos(s1_hidden_state, s2_hidden_state)
      
      # Check cosine similarity 
      if torch.isnan(cos_sim).any():
          nan_indices = torch.where(torch.isnan(cos_sim))[0]
          sentence_ids = batch['sent_ids'][nan_indices]
          print(f"NaN detected in cosine similarity, epoch: {epoch}, batch: {num_batches}, problematic sentence_ids:\n{sentence_ids}")
          continue # Skip this batch
          
      # Use a more numerically stable contrastive loss calculation
      contrastive_loss = torch.tensor(0.0, device=device)
      if a1 > 0 or a2 > 0:  # Only compute if weights are non-zero
          # Add small epsilon to avoid extreme values
          cos_sim = torch.clamp(cos_sim, min=-0.999, max=0.999)
          contrastive_loss = torch.mean((- a1 * labels_float + a2 * (1 - labels_float)) * cos_sim)
          
          # Check contrastive loss
          if torch.isnan(contrastive_loss).any():
              print(f"NaN detected in contrastive loss, epoch: {epoch}, batch: {num_batches}")

      # clear the cache
      s1_ids, s1_mask, s2_ids, s2_mask = None, None, None, None
      s1_hidden_state, s2_hidden_state = None, None
      torch.cuda.empty_cache()

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      
      # Check for NaN in logits
      if torch.isnan(logits).any():
          print(f"WARNING: NaN detected in model output logits, epoch: {epoch}, batch: {num_batches}")
          continue  # Skip this batch
          
      ce_loss = F.cross_entropy(logits, labels, reduction='mean')
      
      # Check for NaN in CE loss
      if torch.isnan(ce_loss).any():
          print(f"WARNING: NaN detected in cross entropy loss, epoch: {epoch}, batch: {num_batches}")
          continue  # Skip this batch
          
      loss = ce_loss + contrastive_loss  # including our regularization term
      
      if torch.isnan(loss).any():
          print(f"WARNING: NaN detected in final loss, epoch: {epoch}, batch: {num_batches}")
          continue  # Skip this batch
          
      loss.backward()
      
      # Add gradient clipping to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

      if num_batches % 200 == 0:
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch} batch {num_batches} / {len(para_train_dataloader)}: train loss :: {loss.item() :.3f}, elapsed time :: {elapsed_time / 60 :.1f}m", flush=True)
        # Print additional diagnostic info
        print(f"  - CE loss: {ce_loss.item():.3f}, Contrastive loss: {contrastive_loss.item():.3f}")

      if num_batches % 400 == 0:
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
        print(f"dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}", flush=True)
        if dev_acc > best_dev_acc:
          best_dev_acc = dev_acc
          save_model(model, optimizer, args, args.ckpt_path)

    train_loss = train_loss / (num_batches if num_batches > 0 else 1)  # Avoid division by zero

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.ckpt_path)

    print(f"\nEpoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}", flush=True)


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.ckpt_path, weights_only=False)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.ckpt_path}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn, num_workers=4)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn, num_workers=4)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-medium')
  parser.add_argument("--a1", type=float, default=1) # weight for the contrastive loss (positive pairs)
  parser.add_argument("--a2", type=float, default=0.1) # weight for the contrastive loss (negative pairs)
  parser.add_argument("--ckpt_path", type=str, default="")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  # print the training hyperparameters
  print(f"Training hyperparameters:")
  print(f"Epochs: {args.epochs} Batch size: {args.batch_size} Learning rate: {args.lr} Model size: {args.model_size}")
  print(f"a1: {args.a1} a2: {args.a2}")
  print(f"Saving checkpoint to {args.ckpt_path}") # NOTE: checkpoint path should be created by the shell script

  print('Started training...', flush=True)
  train(args)
  # print('Started testing...', flush=True)
  # test(args)

import numpy as np
import torch
import random
from tqdm.notebook import tqdm
import BertForMultilabelSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, recall_score, precision_score



def prepare_tensor_dataset(df_sentences, df_labels, num_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    encoded_data = tokenizer.batch_encode_plus(
        df_sentences.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(df_labels.values)
    labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=num_labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset



def train(dataloader_train, num_labels):
    model = BertForMultilabelSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=num_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs = 2
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    for epoch in tqdm(range(1, epochs + 1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        torch.save(model.state_dict(), f'./finetuned_BERT_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

    return model


def evaluate(model, dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def f1_score_func(preds, labels):
    # preds_flat = np.argmax(preds, axis=1).flatten()
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def recall_func(preds, labels):
    # preds_flat = np.argmax(preds, axis=1).flatten()
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    return recall_score(labels_flat, preds_flat, average='weighted')

def precision_func(preds, labels):
    # preds_flat = np.argmax(preds, axis=1).flatten()
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    return precision_score(labels_flat, preds_flat, average='weighted')
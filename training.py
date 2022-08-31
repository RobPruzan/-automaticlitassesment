import random

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

nltk.download('punkt')

df = pd.read_csv('/content/CLEAR_dataset.csv')
print(f'Number of training samples: {df.shape[0]}')

df.sample(100)

df.columns.tolist()

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

excerpts = df.Excerpt.values
targets = df.BT_Easiness.values.astype('float32')

plt.hist(df['BT_Easiness'])
plt.show()

plt.hist(df['Pub Year'])

df.mean(axis=0)

max_len = 0

for i in excerpts:
    input_ids = tokenizer.encode(i, add_special_tokens=True)

    max_len = max(max_len, len(input_ids))

print(max_len)

input_ids = []
attention_masks = []
for i in excerpts:
    encoded_text = tokenizer.encode_plus(
        i,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids.append(encoded_text['input_ids'])

    attention_masks.append(encoded_text['attention_mask'])

len(input_ids[1][0])

attention_masks

input_ids

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(targets)

labels = labels.float()

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 80-20 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 8

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)
#"distilroberta-base"

model = DistilBertForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=1,
    output_attentions=False,
    output_hidden_states=False
)
torch.cuda.empty_cache()
model.cuda()

optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)

EPOCHS = 4

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

training_stats = []
train_loss = []
for epoch in range(EPOCHS):
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        result = model(b_input_ids,

                       attention_mask=b_input_mask,
                       labels=b_labels,
                       return_dict=True
                       )

        loss = result.loss

        train_loss.append(loss.item())

        logits = result.logits

        total_train_loss += loss.item()

        loss = loss.to(torch.float32)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()
        if step % 40 == 0:
            print(f'epoch: {epoch + 1} / {EPOCHS}, step {step + 1} / {len(train_dataloader)}, loss = {loss.item():.4f}')
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'MSE {avg_train_loss:.2f}')
    print("Running Validation...")

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    out_model, label_out = [], []
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(
                b_input_ids,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True,
            )



        loss = loss.to(torch.float32)
        logits = result.logits
        
        for idx, i in enumerate(batch[2]):
          out_model.append(logits[idx].item())
          label_out.append(b_labels[idx].item())
      
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print(f'Validation Loss {avg_val_loss:.2f}')
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'MSE': avg_val_loss,

        }
    )

print("")
print("Training complete!")

def mae(data1, data2):
  total_err = 0
  loops = 0
  for idx, i in enumerate(data1):
    total_err += (abs(i - data2[idx]))
    loops += 1
  return total_err/loops

x = list(range(len(out_model)))
y = out_model

x1 = list(range(len(label_out)))
y1 = label_out



r2 = np.corrcoef(y, y1)[0,1]**2
plt.title(f'r2: {r2} | MAE: {mae(y,y1)}')
plt.scatter(y, y1, label='line 1')

x = list(range(len(train_loss)))
y = train_loss
fig, ax = plt.subplots()

ax.plot(x, y, color='orange', linewidth=2)
ax.grid(False)
plt.xlabel('Step #', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.suptitle('Loss Over Time', fontsize=14, fontweight='bold')
plt.style.use('ggplot')
ax.set_facecolor('w')
fig = plt.gcf()

torch.save(model, '/content/pytorchRoBERTmodel')

PATH = '/content/pytorchRoBERTmodel'
model = torch.load(PATH)
model.eval()
model.to(device)

def predict(text, tokenizer=tokenizer):
    model.eval()
    model.to(device)

    def prepare_data(text, tokenizer):
        input_ids = []
        attention_masks = []

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=315,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return {'input_ids': input_ids, 'attention_masks': attention_masks}

    tokenized_example_text = prepare_data(text, tokenizer)
    with torch.no_grad():
        result = model(
            tokenized_example_text['input_ids'].to(device),
            attention_mask=tokenized_example_text['attention_masks'].to(device),
            return_dict=True
        ).logits

    return result

#Arxiv Abstract

sen = """
Interstellar chemistry is important for galaxy formation, as it determines the rate at which gas can cool, and enables
us to make predictions for observable spectroscopic lines from ions and molecules. We explore two central aspects
of modelling the chemistry of the interstellar medium (ISM): (1) the effects of local stellar radiation, which ionises
and heats the gas, and (2) the depletion of metals onto dust grains, which reduces the abundance of metals in the
gas phase. We run high-resolution (400 M per baryonic particle) simulations of isolated disc galaxies, from dwarfs
to Milky Way-mass, using the fire galaxy formation models together with the chimes non-equilibrium chemistry
and cooling module. In our fiducial model, we couple the chemistry to the stellar fluxes calculated from star particles
using an approximate radiative transfer scheme, and we implement an empirical density-dependent prescription for
metal depletion. For comparison, we also run simulations with a spatially uniform radiation field, and without metal
depletion. Our fiducial model broadly reproduces observed trends in Hi and H2 mass with stellar mass, and in line
luminosity versus star formation rate for [Cii]158µm, [Oi]63µm, [Oiii]88µm, [Nii]122µm and Hα6563˚A. Our simulations
"""

predict(sen)

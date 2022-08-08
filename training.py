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

nltk.download('punkt')

# %matplotlib inline

df = pd.read_csv('/content/train.csv')

print(f'Number of training samples: {df.shape[0]}')

df.sample(100)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

excerpts = df.excerpt.values
targets = df.target.values.astype('float32')

plt.hist(df['target'])
plt.show()

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
        max_length=315,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids.append(encoded_text['input_ids'])

    attention_masks.append(encoded_text['attention_mask'])

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

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
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

torch.save(model, '/content/untitled')

PATH = '/content/pytorchBERTmodel'
model = torch.load(PATH)
model.eval()
model.to(device)


def predict(text, tokenizer):
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


sen = """
Recent JWST observations suggest an excess of 𝑧 & 10 galaxy candidates above most theoretical models. Here, we explore how
the interplay between halo formation timescales, star formation efficiency and dust attenuation affects the properties and number
densities of galaxies we can detect in the early universe. We calculate the theoretical upper limit on the UV luminosity function,
assuming star formation is 100% efficient and all gas in halos is converted into stars, and that galaxies are at the peak age for
UV emission (∼ 10 Myr). This upper limit is ∼ 4 orders of magnitude greater than current observations, implying these are
fully consistent with star formation in ΛCDM cosmology. One day, a woman was walking her two dogs. One was a big, friendly labrador 
and the other was a little yappy dog. As they walked, the little dog started to bark at a cat. The cat hissed and ran away. The 
labrador just stood there wagging his tail. The woman scolded the little dog, "You're supposed to be my protector! Why didn't you 
chase that cat away?" The labrador just looked at her and said, "I'm sorry, but I just don't see the point.
"""
sen_2 = """
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
windows_2 = []
words = word_tokenize(sen_2)
for idx, text in enumerate(words):
    if idx <= len(words) - 21:
        x = ' '.join(words[idx: idx + 20])
        windows_2.append(x)

win_preds_2 = []
for text in windows_2:
    win_preds_2.append(predict(text, tokenizer).item())

windows = []
words = word_tokenize(sen)
for idx, text in enumerate(words):
    if idx <= len(words) - 21:
        x = ' '.join(words[idx: idx + 20])

        windows.append(x)

win_preds = []
for text in windows:
    win_preds.append(predict(text, tokenizer).item())

plt.style.use('seaborn-notebook')
# Data
x = list(range(len(win_preds)))
y = win_preds
x2 = list(range(len(win_preds_2)))
y2 = win_preds_2
# Plot
plt.plot(x, y, color='#ff0000')
plt.plot(x2, y2, color='blue')
plt.grid(color='#cccccc', linestyle='--', linewidth=1)
plt.xlabel('Window Sequence')
plt.ylabel('Difficulty Score')
plt.suptitle('Difficulty Score Over Time', fontsize=14, fontweight='bold')
plt.show()

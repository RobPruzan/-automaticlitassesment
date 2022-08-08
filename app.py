import statistics
import string

import gensim.downloader as api
import gradio as gr
import nltk
import pandas as pd
import readability
import seaborn as sns
import torch
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer
from transformers import pipeline

nltk.download('cmudict')

nltk.download('stopwords')

nltk.download('punkt')

glove_vectors = api.load('glove-wiki-gigaword-100')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# loading model
PATH = 'pytorchBERTmodel'
model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

model.to('cpu')

p = pipeline("automatic-speech-recognition")

w2v = dict({})
for idx, key in enumerate(glove_vectors.key_to_index.keys()):
    w2v[key] = glove_vectors.get_vector(key)


def calculate_diversity(text):
    stop_words = set(stopwords.words('english'))
    for i in string.punctuation:
        stop_words.add(i)

    tokenized_text = word_tokenize(text)
    tokenized_text = list(map(lambda word: word.lower(), tokenized_text))
    sim_words = {}
    if len(tokenized_text) <= 1:
        return 1, "More Text Required"

    for idx, anc_word in enumerate(tokenized_text):
        if anc_word in stop_words:
            continue
        if idx in sim_words:
            sim_words[idx] = sim_words[idx]
            continue

        vocab = [anc_word]

        for pos, comp_word in enumerate(tokenized_text):

            try:
                if not comp_word in stop_words and cosine_similarity(w2v[anc_word].reshape(1, -1),
                                                                     w2v[comp_word].reshape(1, -1)) > .75:
                    vocab.append(comp_word)

                sim_words[idx] = vocab

            except KeyError:
                continue

    scores = {}
    for k, value in sim_words.items():
        if len(value) == 1:
            scores[k] = 1
            continue

        t_sim = len(value) - 1
        t_rep = (len(value) - 1) - (len(set(value)))

        score = ((t_sim - t_rep) / t_sim) ** 2

        scores[key] = score

    mean_score = 0
    total = 0

    for value in scores.values():
        mean_score += value
        total += 1

    return scores, mean_score / total


def dict_to_list(dictionary, max_size=10):
    outer_list = []
    inner_list = []

    for value in dictionary.values():
        inner_list.append(value)
        if len(inner_list) == max_size:
            outer_list.append(inner_list)
            inner_list = []
    if len(inner_list) > 0:
        outer_list.append(inner_list)
    return outer_list


def heatmap(scores, df):
    total = 0
    loops = 0

    for ratio in scores.values():
        # conditional to visualize the difference between no ratio and a 0 ratio score
        if ratio != -.3:
            total += ratio
            loops += 1

    diversity_average = total / loops

    return sns.heatmap(df, cmap='gist_gray_r', vmin=-.3).set(
        title='Word Diversity Score Heatmap (Average Score: ' + str(diversity_average) + ')')


def stats(text):
    results = readability.getmeasures(text, lang='en')
    return results


def predict(text, tokenizer=tokenizer):
    model.eval()
    model.to('cpu')

    def prepare_data(text, tokenizer):
        input_ids = []
        attention_masks = []

        encoded_text = tokenizer.encode_plus(
            text,
            truncation=True,
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
        return {'input_ids': input_ids, 'attention_masks': attention_masks}

    tokenized_example_text = prepare_data(text, tokenizer)
    with torch.no_grad():
        result = model(
            tokenized_example_text['input_ids'].to('cpu'),
            attention_mask=tokenized_example_text['attention_masks'].to('cpu'),
            return_dict=True
        ).logits

    return result


def reading_difficulty(excerpt):
    if len(excerpt) == 0:
        return "No Text Provided"
    windows = []
    words = tokenizer.tokenize(excerpt)

    if len(words) > 301:
        for idx, text in enumerate(words):
            if idx % 300 == 0:
                if idx <= len(words) - 301:
                    x = ' '.join(words[idx: idx + 299])
                    windows.append(x)

        win_preds = []
        for text in windows:
            win_preds.append(predict(text, tokenizer).item())
        result = statistics.mean(win_preds)
        score = -(result * 1.786 + 6.4) + 10
        return score

    else:
        result = predict(excerpt).item()
        score = -(result * 1.786 + 6.4) + 10
        return score


def calculate_stats(file_name, data_index):
    # unicode escape only for essays
    with open(file_name, encoding='unicode_escape') as f:
        information = {'lines': 0, 'words_per_sentence': 0, 'words': 0, 'syll_per_word': 0, 'characters_per_word': 0,
                       'reading_difficulty': 0}
        reader = csv.reader(f)

        for line in reader:

            if len(line[data_index]) < 100:
                continue

            # if detect(line[data_index][len(line[data_index]) -400: len(line[data_index])-1]) == 'en':

            try:
                stat = stats(line[data_index])

            except ValueError:
                continue

            information['lines'] += 1
            print(information['lines'])
            information['words_per_sentence'] += stat['sentence info']['words_per_sentence']
            information['words'] += stat['sentence info']['words']
            information['syll_per_word'] += stat['sentence info']['syll_per_word']
            information['characters_per_word'] += stat['sentence info']['characters_per_word']
            information['reading_difficulty'] += reading_difficulty(line[data_index])

    for i in information:
        if i != 'lines' and i != 'words':
            information[i] /= information['lines']

    return information


def transcribe(audio):
    # speech to text using pipeline
    text = p(audio)["text"]
    transcription.append(text)
    return text


def compute_score(target, actual):
    target = target.lower()
    actual = actual.lower()
    return fuzz.ratio(target, actual)


def phon(text):
    alph = nltk.corpus.cmudict.dict()
    text = word_tokenize(text)
    pronun = []
    for word in text:
        try:
            pronun.append(alph[word][0])
        except Exception as e:
            pronun.append(word)
    return pronun


def gradio_fn(text, audio, target, actual_audio):
    if text is None and audio is None and target is None and actual_audio is None:
        return "No Inputs", "No Inputs", "No Inputs", "No Inputs"
    speech_score = 0
    div = calculate_diversity(text)

    if actual_audio is not None:
        actual = p(actual_audio)["text"]
        print('sdfgs')
        speech_score = compute_score(target, actual)

        return "Difficulty Score: " + str(reading_difficulty(actual)), "Transcript: " + str(
            actual.lower()), "Diversity Score: " + str(div[1]), "Speech Score: " + str(speech_score)

    transcription = []
    if audio is not None:
        text = p(audio)["text"]
        transcription.append(text)
        state = div[0]
        return "Difficulty Score: " + str(reading_difficulty(text)), "Transcript: " + str(
            transcription[-1].lower()), "Diversity Score: " + str(div[1]), "No Inputs"

    return "Difficulty Score: " + str(reading_difficulty(text)), "Diversity Score: " + str(
        div[1]), "No Audio Provided", "No Inputs"


def plot():
    text = state
    diversity = calculate_diversity(text)[0]
    print(diversity)
    df = pd.DataFrame(dict_to_list(diversity))
    return heatmap(diversity, df)


import csv

example_data = []
x = 0
with open('train.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        example_data.append([line[3]])
        x += 1
        if x > 100:
            break

state = {}
interface = gr.Interface(
    fn=gradio_fn,
    inputs=[gr.components.Textbox(
        label="Text"),
        gr.components.Audio(
            label="Speech Translation",
            source="microphone",
            type="filepath"),
        gr.components.Textbox(
            label="Target Text to Recite"
        ),
        gr.components.Audio(
            label="Read Text Above for Score",
            source="microphone",
            type="filepath")
    ],

    outputs=["text", "text", "text", "text"],
    theme="huggingface",
    description="Enter text or speak into your microphone to have your text analyzed!",

    rounded=True,
    container=True

).launch()

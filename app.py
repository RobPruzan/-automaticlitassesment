import csv
import string
import json

import gensim.downloader as api
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import gradio as gr
import readability
import seaborn as sns
import torch
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer
from transformers import pipeline


nltk.download('wordnet')

nltk.download('omw-1.4')

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


def wn_syns(word):
    synonyms = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return set(synonyms)


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

    for idx, anc in enumerate(tokenized_text):
        if anc in stop_words or not anc in w2v or anc.isdigit():
            sim_words[idx] = '@'
            continue

        vocab = [anc]

        for pos, comp in enumerate(tokenized_text):
            if pos == idx:
                continue
            if comp in stop_words:
                continue
            if not comp.isalpha():
                continue
            try:
                if cosine_similarity(w2v[anc].reshape(1, -1), w2v[comp].reshape(1, -1)) > .7 or comp in wn_syns(anc):
                    vocab.append(comp)
            except KeyError:
                continue
        sim_words[idx] = vocab
    scores = {}
    for key, value in sim_words.items():
        if len(value) == 1:
            scores[key] = -1
            continue
        t_sim = len(value)
        t_rep = (len(value)) - (len(set(value)))

        score = ((t_sim - t_rep) / t_sim) ** 2

        scores[key] = score

    mean_score = 0
    total = 0

    for value in scores.values():
        if value == -1:
            continue
        mean_score += value
        total += 1
    try:
        return scores, {"Diversity Score": mean_score / total}
    except ZeroDivisionError:
        return scores, {"Dviersity Score": "Not Enough Data"}


def get_scores(text):
    return calculate_diversity(text)[0]


def get_mean_score(text):
    return calculate_diversity(text)[1]


def dict_to_list(dictionary, max_size=10):
    outer_list = []
    inner_list = []

    for key, value in dictionary.items():
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


def level(score):
    if score <= 3:
        return "n Elementary School"
    elif 3 <= score <= 6:
        return " Middle School"
    elif 6 <= score <= 8:
        return " High School"
    else:
        return " College"


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
        result = np.mean(win_preds)
        score = -(result * 1.786 + 6.4) + 10
        return f'Difficulty Level: {round(score,2)}/10 | A {level(score)} student could understand this'

    else:
        result = predict(excerpt).item()
        score = -(result * 1.786 + 6.4) + 10
        return f'Difficulty Level: {round(score,2)}/10 | A {level(score)} student could understand this'"

def calculate_stats(file_name, data_index):
    with open(file_name, encoding='unicode_escape') as f:
        information = {'lines': 0, 'words_per_sentence': 0, 'words': 0, 'syll_per_word': 0, 'characters_per_word': 0,
                       'reading_difficulty': 0}
        reader = csv.reader(f)

        for line in reader:

            if len(line[data_index]) < 100:
                continue

            try:
                stat = stats(line[data_index])

            except ValueError:
                continue

            information['lines'] += 1
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

    def remove_digits(lists):
        for lst in lists:
            for idx, word in enumerate(lst):
                lst[idx] = ''.join([letter for letter in word if not letter.isdigit()])
        return lists

    output = []
    for i in remove_digits(pronun):
        output.append('-'.join(i).lower())
    return '  '.join(output)


def plot():
    diversity = calculate_diversity(text)[0]
    print(diversity)
    df = pd.DataFrame(dict_to_list(diversity))
    return heatmap(diversity, df)


def diversity_inter(text):
    words = word_tokenize(text)
    scores = get_scores(text)
    interpret_values = [('', 0.0)]
    for key, value in scores.items():
        interpret_values.append((words[key], value))
    interpret_values.append(('', 0.0))
    return {'original': text, 'interpretation': interpret_values}


def sliding_window(text):
    words = word_tokenize(text)
    improved_window = []
    improved_wind_preds = []
    for idx, text in enumerate(words):
        if idx <= len(words) - 26:
            x = ' '.join(words[idx: idx + 25])
            throw_away = []
            score = 0
            for idx, i in enumerate(range(idx, idx + 25)):
                if idx == 0:
                    better_prediction = -(predict(x).item() * 1.786 + 6.4) + 10
                    score = better_prediction
                    throw_away.append((better_prediction, i))
                else:
                    throw_away.append((score, i))

            improved_window.append(throw_away)
    average_scores = {k: 0 for k in range(len(words) - 1)}
    total_windows = {k: 0 for k in range(len(words) - 1)}
    for idx, i in enumerate(improved_window):
        for score, idx in i:
            average_scores[idx] += score
            total_windows[idx] += 1

    for k, v in total_windows.items():
        if v != 0:
            average_scores[k] /= v

    inter_scores = [v for v in average_scores.values()]
    copy_list = inter_scores.copy()
    while len(inter_scores) <= len(words) - 1:
        inter_scores.append(copy_list[-1])

    x = list(range(len(inter_scores)))
    y = inter_scores

    fig, ax = plt.subplots()

    ax.plot(x, y, color='orange', linewidth=2)
    ax.grid(False)
    plt.xlabel('Word Number', fontweight='bold')
    plt.ylabel('Difficulty Score', fontweight='bold')
    fig.patch.set_facecolor('white')
    plt.suptitle('Difficulty Score Across Text', fontsize=14, fontweight='bold')
    plt.style.use('ggplot')

    fig = plt.gcf()

    map = [('', 0)]
    maxy = max(inter_scores)
    miny = min(inter_scores)
    spread = maxy - miny

    for idx, i in enumerate(words):
        map.append((i, (inter_scores[idx] - miny) / spread))
    map.append(('', 0))

    return fig, map



def get_plot(text):
    return sliding_window(text)[0]


def get_dif_inter(text):
    return {'original': text, 'interpretation': sliding_window(text)[1]}


def speech_to_text(speech, target):
    text = p(speech)["text"]
    return text.lower(), {'Pronunciation Score': compute_score(text, target) / 100}, phon(target)


def my_i_func(text):
    return {"original": "", "interpretation": [('', 0.0), ('what', -0.2), ('great', 0.3), ('day', 0.5), ('', 0.0)]}


def gen_syns(word, level):
  with open('balanced_synonym_data.json') as f:
    word = word.strip(" ")
    data = json.loads(f.read())
    school_to_level = {"Elementary Level":'1', "Middle School Level":'2', "High School Level":'3', "College Level":'4'}
    pins = wn_syns(word)
    reko = []
    for i in pins:
      if i in data[school_to_level[level]]:
        reko.append(i)
    str_reko = ""
    for idx, i in enumerate(reko):
      if idx != len(reko) -1:
        str_reko+= i + ' | '
      else:
        str_reko+= i
    return str_reko

with gr.Blocks(title="Automatic Literacy and Speech Assesmen") as demo:
  gr.HTML("""<center><h7 style="font-size: 35px">Automatic Literacy and Speech Assesment</h7></center>""")
  with gr.Column():
    with gr.Row():
      with gr.Box():

        with gr.Column():
          with gr.Group():
            with gr.Tabs():
              
                with gr.TabItem("Text"):
                    in_text = gr.Textbox(label="Input Text Or Speech For Analysis")
                    grade = gr.Button("Grade Your Text")
                with gr.TabItem("Speech"):
                    audio_file = gr.Audio(source="microphone",type="filepath")
                    grade1 = gr.Button("Grade Your Speech")
            with gr.Group():     
              gr.Markdown("Reading Level Based Synonyms | Enter only one word at a time")
              words = gr.Textbox(label="Word For Synonyms")
              lvl = gr.Dropdown(choices=["Elementary Level", "Middle School Level", "High School Level", "College Level" ], label="Intended Reading Level For Synonym")
              get_syns = gr.Button("Get Synonyms")
              reccos = gr.Label()
              

      with gr.Box():
          diff_output = gr.Label(label='Difficulty Level',show_label=True)
          gr.Markdown("Diversity Score Across Text")
          plotter = gr.Plot()




    with gr.Row():
      with gr.Box():
        div_output = gr.Label(label='Diversity Score', show_label=False)
        gr.Markdown("Diversity Heatmap | Blue cells are omitted from score | Darker = More Diverse")
        interpretation = gr.components.Interpretation(in_text, label="Diversity Heatmap")
      with gr.Box():
          gr.Markdown("Relative Difficulty Heatmap- How confusing the text is in that area") 
          interpretation2 = gr.components.Interpretation(in_text, label="Difficulty Heatmap")
  with gr.Row():
    with gr.Box():
      with gr.Group():      
        target = gr.Textbox(label="Target Text")
      with gr.Group():      
        audio_file1 = gr.Audio(source="microphone",type="filepath")
        b1 = gr.Button("Grade Your Pronunciation")
    with gr.Box():
      some_val = gr.Label()
      text = gr.Textbox()
      phones = gr.Textbox()
        
  gr.Markdown("""**Reading Difficulty**-  Automatically determining how difficult something is to read is a difficult task as underlying 
                 semantics are relevant. To efficiently compute text difficulty, a Distil-Bert pre-trained model is fine-tuned for regression 
                 using The CommonLit Ease of Readability (CLEAR) Corpus. This model scores the text on how difficult it would be for a student
                 to understand.
              """)
  gr.Markdown("""**Lexical Diversity**-  The lexical diversity score is computed by taking the ratio of unique similar words to total similar words 
                  squared. The similarity is computed as if the cosine similarity of the word2vec embeddings is greater than .75. It is bad writing/speech 
                  practice to repeat the same words when it's possible not to. Vocabulary diversity is generally computed by taking the ratio of unique 
                  strings/ total strings. This does not give an indication if the person has a large vocabulary or if the topic does not require a diverse 
                  vocabulary to express it. This algorithm only scores the text based on how many times a unique word was chosen for a semantic idea, e.g., 
                  "Forest" and "Trees" are 2 words to represent one semantic idea, so this would receive a 100% lexical diversity score, vs using the word
                  "Forest" twice would yield you a 25% diversity score, (1 unique word/ 2 total words)^2
              """)
  gr.Markdown("""**Speech Pronunciation Scoring-**-  The Wave2Vec 2.0 model is utilized to convert audio into text in real-time. The model predicts words or phonemes
                  (smallest unit of speech distinguishing one word (or word element) from another) from the input audio from the user. Due to the nature of the model, 
                  users with poor pronunciation get inaccurate results. This project attempts to score pronunciation by asking a user to read a target excerpt into the 
                  microphone. We then pass this audio through Wave2Vec to get the inferred intended words. We measure the loss as the Levenshtein distance between the 
                  target and actual transcripts- the Levenshtein distance between two words is the minimum number of single-character edits required to change one word 
                  into the other.
              """)


  grade.click(reading_difficulty, inputs=in_text, outputs=diff_output)
  grade.click(get_mean_score, inputs=in_text, outputs=div_output)
  grade.click(diversity_inter, inputs=in_text, outputs=interpretation)
  grade.click(get_dif_inter, inputs=in_text, outputs=interpretation2)
  grade.click(get_plot, inputs=in_text, outputs=plotter)
  grade1.click(speech_to_score, inputs=audio_file, outputs=diff_output)
  b1.click(speech_to_text, inputs=[audio_file1, target], outputs=[text, some_val, phones])
  get_syns.click(gen_syns, inputs=[words, lvl], outputs=reccos)
demo.launch(debug=True)


import csv
import string
import json
import sys
import logging
import argparse

import gensim.downloader as api
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import gradio as gr
import readability
import seaborn as sns
import torch
import torch.nn.functional as F
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer
from transformers import pipeline
from transformers import BertTokenizer
from transformers import AutoTokenizer, BertForSequenceClassification


nltk.download('wordnet')

nltk.download('omw-1.4')

nltk.download('cmudict')

nltk.download('stopwords')

nltk.download('punkt')

glove_vectors = api.load('glove-wiki-gigaword-100')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# loading model
PATH = 'pytorchRoBERTmodel'
model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

model.to('cpu')

p = pipeline("automatic-speech-recognition")

with open('balanced_synonym_data.json') as f:
  data = json.loads(f.read())
  
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
    if len(tokenized_text) > 1500:
      return f'Input length of:{len(tokenized_text)} exceed limit of 1500 tokens'
      
    tokenized_text = list(map(lambda word: word.lower(), tokenized_text))
    global sim_words
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
                if cosine_similarity(w2v[anc].reshape(1, -1), w2v[comp].reshape(1, -1)) > .75 or comp in wn_syns(anc):
                    vocab.append(comp)
            except KeyError:
                continue
        sim_words[idx] = vocab
    print(sim_words)
    scores = {}
    for key, value in sim_words.items():
        if len(value) == 1:
            scores[key] = -1
            continue
        t_sim = len(value)
        t_rep = (len(value)) - (len(set(value)))

        score = (t_sim - t_rep) / t_sim

        scores[key] = score

    mean_score = 0
    total = 0
    
    for value in scores.values():
        if value == -1:
            continue
        mean_score += value
        total += 1
        words = word_tokenize(text)

    interpret_values = [('', 0.0)]

    for key, value in scores.items():
        interpret_values.append((words[key], value))

    interpret_values.append(('', 0.0))
    print(interpret_values)
    int_vals = {'original': text, 'interpretation': interpret_values}
    try:

        return int_vals, {"Diversity Score": mean_score / total}
    except ZeroDivisionError:

        return int_vals, {"Dviersity Score": "Not Enough Data"}

def get_sim_words(text, word):
    word = word.strip()
    index = 0
    text = word_tokenize(text)
    print(sim_words)
    for idx, i in enumerate(text):
        if word == i:
            index = idx
            break
    return ', '.join(sim_words[index])


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
        return "Difficulty Level: " + str(round(score, 2)) + '/10' + ' | A' + str(
            level(score)) + " student could understand this"

    else:
        result = predict(excerpt).item()
        score = -(result * 1.786 + 6.4) + 10
        return 'Difficulty Level: ' + str(round(score, 2)) + '/10' + ' | A' + str(
            level(score)) + " student could understand this"


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
    print(target)
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
    print(inter_scores)
    while len(inter_scores) <= len(words) - 1:
        inter_scores.append(copy_list[-1])

    x = list(range(len(inter_scores)))
    y = inter_scores

    fig, ax = plt.subplots()

    ax.plot(x, y, color='orange', linewidth=2)
    ax.grid(False)
    plt.xlabel('Word Number', fontweight='bold')
    plt.ylabel('Difficulty Score', fontweight='bold')
    plt.suptitle('Difficulty Score Across Text', fontsize=14, fontweight='bold')
    plt.style.use('ggplot')
    ax.set_facecolor('w')
    fig = plt.gcf()

    mapd = [('', 0)]
    maxy = max(inter_scores)
    miny = min(inter_scores)
    spread = maxy - miny

    for idx, i in enumerate(words):
        mapd.append((i, (inter_scores[idx] - miny) / spread))
    mapd.append(('', 0))

    return fig, {'original': text, 'interpretation': mapd}

def speech_to_text(speech, target):
    text = p(speech)["text"]
    return text.lower(), {'Pronunciation Score': compute_score(text, target) / 100}, phon(target)
    
def speech_to_score(speech):
    text = p(speech)["text"]
    return reading_difficulty(text), text

def my_i_func(text):
    return {"original": "", "interpretation": [('', 0.0), ('what', -0.2), ('great', 0.3), ('day', 0.5), ('', 0.0)]}

def gen_syns(word, level):
    word = word.strip(" ")
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

def get_level(word):
  with open('balanced_synonym_data.json') as f:
    word = word.strip(" ")
    data = json.loads(f.read())
    level = 0
    for k, v in data.items():
      if word in v:
        level = k
    if level == 0:
      return -1
    return level

def vocab_level_inter(text):
  text = word_tokenize(text)
  interp = [('',0)]
  sum = 0
  total = 0
  for idx, i in enumerate(text):
    lvl = int(get_level(i))/4
    interp.append((i, lvl))
    sum+= lvl
    total += 1
  interp.append(('', 0))
  return {'original': text, 'interpretation': interp}, f'{level(sum/total*4*2.5)[1:]} Level Vocabulary'



logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
tokenizer4 = AutoTokenizer.from_pretrained('kanishka/GlossBERT')

def construct_context_gloss_pairs_through_nltk(input, target_start_id, target_end_id):
    """
    construct context gloss pairs like sent_cls_ws
    :param input: str, a sentence
    :param target_start_id: int
    :param target_end_id: int
    :param lemma: lemma of the target word
    :return: candidate lists
    """
    
    sent = tokenizer4.tokenize(input)
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])
    if len(sent) > target_end_id:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"'] + sent[target_end_id:]
    else:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"']

    sent = " ".join(sent)

    candidate = []
    syns = wn.synsets(target)
    
    for syn in syns:
        if target == syn.name().split('.')[0]:
          continue
        
        gloss = (syn.definition(), syn.name())
        candidate.append((sent, f"{target} : {gloss}", target, gloss))

    assert len(candidate) != 0, f'there is no candidate sense of "{target}" in WordNet, please check'
    # print(f'there are {len(candidate)} candidate senses of "{target}"')


    return candidate


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_to_features(candidate, tokenizer3, max_seq_length=512):

    candidate_results = []
    features = []
    for item in candidate:
        text_a = item[0] # sentence
        text_b = item[1] # gloss
        candidate_results.append((item[-2], item[-1])) # (target, gloss)


        tokens_a = tokenizer3.tokenize(text_a)
        tokens_b = tokenizer3.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer3.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))


    return features, candidate_results



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def infer(input, target_start_id, target_end_id, args):
    sent = tokenizer4.tokenize(input)
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    label_list = ["0", "1"]
    num_labels = len(label_list)
    
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          num_labels=num_labels)
    model.to(device)

    # print(f"input: {input}\ntarget: {target}")
    examples = construct_context_gloss_pairs_through_nltk(input, target_start_id, target_end_id)
    eval_features, candidate_results = convert_to_features(examples, tokenizer4)
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)


    model.eval()
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None).logits
    logits_ = F.softmax(logits, dim=-1)
    logits_ = logits_.detach().cpu().numpy()
    output = np.argmax(logits_, axis=0)[1]
    results= []
    for idx, i in enumerate(logits_):
      results.append((candidate_results[idx][1], i[1]*100))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    return sorted_results

def format_for_gradio(inp):
  retval = ''
  for idx, i in enumerate(inp):
    if idx == len(inp)-1:
      retval += i.split('.')[0]
      break
    retval += f'''{i.split('.')[0]} | '''
  return retval


def smart_synonyms(text, level):
  parser = argparse.ArgumentParser()
  parser.add_argument("--bert_model", default="kanishka/GlossBERT", type=str)
  parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
  args, unknown = parser.parse_known_args()

  location = 0
  word = ''
  tokens = tokenizer4.tokenize(text)
  school_to_level = {"Elementary Level":'1', "Middle School Level":'2', "High School Level":'3', "College Level":'4'}
  for idx, i in enumerate(tokens):
    if i[0] == '@':
      location = idx
      text = text.replace('@', '')
      word = tokens[location]
      break 
  raw_syns = []
  raw_defs = []
  raw_scores = []
  syns = []
  defs = []
  scores = []
  preds = infer(text, location, location+1, args)
  for i in preds:
    if not i[0][1].split('.')[0] in data[school_to_level[level]]:
      continue
    raw_syns.append(i[0][1])
    raw_defs.append(i[0][0])
    raw_scores.append(i[1])
    if i[1] > 5:
      syns.append(i[0][1])
      defs.append(i[0][0])
      scores.append(i[1])

  if not syns:
    top_syns = int(len(raw_syns)*.25//1+1)
    syns = raw_syns[:top_syns]
    defs = raw_defs[:top_syns]
    scores = raw_scores[:top_syns]

  cleaned_syns = format_for_gradio(syns)
  cleaend_defs = format_for_gradio(defs)
  
  return f'{cleaned_syns}: Definition- {cleaend_defs} | '



with gr.Blocks(title="Automatic Literacy and Speech Assesment") as demo:
  gr.HTML("""<center><h7 style="font-size: 35px">Automatic Literacy and Speech Assesment</h7></center>""")
  gr.HTML("""<center><h7 style="font-size: 15px">This may take 60s to generate all statistics</h7></center>""")
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
              gr.Markdown("""Reading Level Based Synonyms | Enter a sentence with the word you want a synonym | Add an @ before the target word for synonym, e.g. - "Today is an @amazing day"- target word = amazing" """)
              words = gr.Textbox(label="Text with word for synonyms")
              lvl = gr.Dropdown(choices=["Elementary Level", "Middle School Level", "High School Level", "College Level" ], label="Intended Reading Level For Synonym")
              get_syns = gr.Button("Get Synonyms")
              reccos = gr.Label()
              

      with gr.Box():
          diff_output = gr.Label(label='Difficulty Level',show_label=True)
          gr.Markdown("Difficulty Score Across Text")
          plotter = gr.Plot()




    with gr.Row():
      with gr.Box():
        div_output = gr.Label(label='Diversity Score', show_label=False)
        gr.Markdown("Diversity Heatmap | Blue cells are omitted from score. | Darker = More Diverse")
        interpretation = gr.components.Interpretation(in_text, label="Diversity Heatmap")
        
        gr.Markdown("Find Similar Words | Word must be part of analysis text box | Enter only one word at a time")
        words1 = gr.Textbox(label="Word For Similarity")
        find_sim = gr.Button("Find Similar Words")
        sims = gr.Label()
      with gr.Box():
        gr.Markdown("Relative Difficulty Heatmap- How confusing the text is in that area of text") 
        interpretation2 = gr.components.Interpretation(in_text, label="Difficulty Heatmap")
      with gr.Box():
        vocab_output = gr.Label(label='Vocabulary Level', show_label=True)
        gr.Markdown("Vocabulary Level Heatmap | Darker = Higher Level | Blue cells are not in vocabulary")
        interpretation3 = gr.components.Interpretation(in_text, label="Interpretation of Text")
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
                  . The similarity is computed as if the cosine similarity of the word2vec embeddings is greater than .75. It is bad writing/speech 
                  practice to repeat the same words when it's possible not to. Vocabulary diversity is generally computed by taking the ratio of unique 
                  strings/ total strings. This does not give an indication if the person has a large vocabulary or if the topic does not require a diverse 
                  vocabulary to express it. This algorithm only scores the text based on how many times a unique word was chosen for a semantic idea, e.g., 
                  "Forest" and "Woods" are 2 words to represent one semantic idea, so this would receive a 100% lexical diversity score, vs using the word
                  "Forest" twice would yield you a 25% diversity score, (1 unique word/ 2 total words)
              """)
  gr.Markdown("""**Speech Pronunciation Scoring-**-  The Wave2Vec 2.0 model is utilized to convert audio into text in real-time. The model predicts words or phonemes
                  (smallest unit of speech distinguishing one word (or word element) from another) from the input audio from the user. Due to the nature of the model, 
                  users with poor pronunciation get inaccurate results. This project attempts to score pronunciation by asking a user to read a target excerpt into the 
                  microphone. We then pass this audio through Wave2Vec to get the inferred intended words. We measure the loss as the Levenshtein distance between the 
                  target and actual transcripts- the Levenshtein distance between two words is the minimum number of single-character edits required to change one word 
                  into the other.
              """)


  grade.click(reading_difficulty, inputs=in_text, outputs=diff_output)
  grade.click(calculate_diversity, inputs=in_text, outputs=[interpretation, div_output])
  grade.click(sliding_window, inputs=in_text, outputs=[plotter, interpretation2])
  grade.click(vocab_level_inter, inputs=in_text, outputs=[interpretation3, vocab_output])
  grade1.click(speech_to_score, inputs=audio_file, outputs=diff_output)
  b1.click(speech_to_text, inputs=[audio_file1, target], outputs=[text, some_val, phones])
  get_syns.click(smart_synonyms, inputs=[words, lvl], outputs=reccos)
  find_sim.click(get_sim_words, inputs=[in_text, words1], outputs=sims)
demo.launch(debug=True)


from flask import *
import io
from werkzeug.utils import secure_filename
from flaskext.markdown import Markdown

import stanza

import textrazor
from transformers import *
import torch

import spacy
from spacy import displacy

from flair.data import Sentence
from flair.models import SequenceTagger

import os
import pandas as pd

app = Flask(__name__)
app.config.update(DEBUG=True)
Markdown(app)

# STANZA
stanza_en = stanza.Pipeline('en')
stanza_fr = stanza.Pipeline('fr')

# SPACY
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

# TEXTRAZOR
textrazor.api_key = "9fb66f17ec9ab7a7f5015b68f27be5ed02a81c7b49dd2c607f2ebea2"

# TRANSFORMERS
transformers_nlp = pipeline('ner')

# FLAIR
flair_en = SequenceTagger.load('ner')
flair_fr = SequenceTagger.load('fr-ner')

#####################################################
############### BENCHMARK 
#####################################################
@app.route('/', methods = ['GET', 'POST'])
def home() :

    if request.method == 'POST' :
        
        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            stanza_nlp = stanza_en
            spacy_nlp = spacy_en
            flair_nlp = flair_en
        else :
            stanza_nlp = stanza_fr
            spacy_nlp = spacy_fr
            flair_nlp = flair_fr

        # Lists for benchmark dataframe
        libs = ['Stanza', 'Spacy', 'Text Razor', 'Transfromers', 'Flair']
        entities_list = []
        token_counter = []
        sentences_counter = []
        entities_counter = []

        # STANZA
        stanza_doc = stanza_nlp(text)
        # Document conversion to dict for manipulation
        stanza_result = stanza_doc.to_dict()
        stanza_text_list = []

        # We loop inside result dict to append lists
        for sentence in range(len(stanza_result)) :
            for word in range(len(stanza_result[sentence])) :
                stanza_text_list.append(stanza_result[sentence][word]['text'])

        # Entities
        stanza_ner_result = stanza_doc.entities
        ent_list = []

        for entity in range(len(stanza_ner_result)) :
            ent_text = stanza_ner_result[entity].text
            ent_type = stanza_ner_result[entity].type 
            ent_list.append(f'({ent_text}, {ent_type})')
        
        # Append Stanza counters
        token_counter.append(len(stanza_text_list))
        sentences_counter.append(len(stanza_doc.sentences))
        entities_counter.append(len(ent_list))
        entities_list.append(ent_list)

        # SPACY
        spacy_doc = spacy_nlp(text)
        # Token texts
        spacy_text_list = [token.text for token in spacy_doc]
        spacy_sent = [sent.text for sent in spacy_doc.sents]
        spacy_entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]

        # Append Spacy counters
        token_counter.append(len(spacy_text_list))
        sentences_counter.append(len(spacy_sent))
        entities_counter.append(len(spacy_entities))
        entities_list.append(spacy_entities)

        # TEXTRAZOR
        client_textrazor = textrazor.TextRazor(extractors=["entities", "topics"])
        response_textrazor = client_textrazor.analyze(text)

        razor_entity_list = []

        for entity in response_textrazor.entities():
            razor_ent = entity.id
            razor_type = entity.dbpedia_types
            razor_entity_list.append(f'({razor_ent}, {razor_type})')

        # Append Textrazor counters
        token_counter.append(len(spacy_text_list))
        sentences_counter.append(len(spacy_sent))
        entities_counter.append(len(razor_entity_list))
        entities_list.append(razor_entity_list)

        # TRANSFORMERS
        transformers_entities = transformers_nlp(text)
        transformers_entities_list = []
        for ent in range(len(transformers_entities)) :
            trans_ent = transformers_entities[ent]['word']
            trans_type = transformers_entities[ent]['entity']
            transformers_entities_list.append(f'({trans_ent}, {trans_type})')

        # Append Transformers XL counters
        token_counter.append(len(spacy_text_list))
        sentences_counter.append(len(spacy_sent))
        entities_counter.append(len(transformers_entities_list))
        entities_list.append(transformers_entities_list)

        # FLAIR
        flair_sentence = Sentence(text)

        # Run NER on sentence to identify Entities
        flair_nlp.predict(flair_sentence)

        flair_entities_list = []
        # print the entities with below command
        for entity in flair_sentence.get_spans('ner'):
            entities = entity.to_dict()
            entity_name = entities['text']
            entity_type = entities['labels']
            flair_entities_list.append(f'({entity_name}, {entity_type})')

        # Append Transformers XL counters
        token_counter.append(len(flair_sentence.tokens))
        sentences_counter.append(len(spacy_sent))
        entities_counter.append(len(flair_entities_list))
        entities_list.append(flair_entities_list)

        # We display result in a dataframe
        benchmark_result_df = pd.DataFrame({
            'Library' : libs,
            'Number of Tokens' : token_counter,
            'Number of Sentences' : sentences_counter,
            'Number of Entities' : entities_counter,
            'Entities' : entities_list})

        return render_template('index.html', text= text, dataset = [benchmark_result_df.to_html(classes= 'data')])

    return render_template('index.html')

#####################################################
############### STANZA 
#####################################################
@app.route('/stanza_page', methods = ['GET', 'POST'])
def stanza_page() :

    if request.method == 'POST' :

        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            stanza_nlp = stanza_en
        else :
            stanza_nlp = stanza_fr

        # Return tokenization as Document object
        stanza_doc = stanza_nlp(text)

        # Document conversion to dict for manipulation
        stanza_result = stanza_doc.to_dict()

        # Lists preparation for DataFrame
        text_list = []
        lemma_list = []
        upos_list = []

        # We loop inside result dict to append lists
        for sentence in range(len(stanza_result)) :
            for word in range(len(stanza_result[sentence])) :
                text_list.append(stanza_result[sentence][word]['text'])
                lemma_list.append(stanza_result[sentence][word]['lemma'])
                upos_list.append(stanza_result[sentence][word]['upos'])

        # We display result in a dataframe
        stanza_result_df = pd.DataFrame({
            'Text' : text_list,
            'Lemma' : lemma_list,
            'Part Of Speech (pos)' : upos_list})

        # Return entities as JSON object
        stanza_ner_result = stanza_doc.entities
        # Results format
        ent_text_list = []
        ent_type_list = []

        for entity in range(len(stanza_ner_result)) :
            ent_text_list.append(stanza_ner_result[entity].text)
            ent_type_list.append(stanza_ner_result[entity].type)

        df_entities = pd.DataFrame({
            'Text' : ent_text_list,
            'Type' : ent_type_list})

        # Counters
        sentences_counter = len(stanza_doc.sentences)
        token_counter = len(text_list)
        entities_counter = len(stanza_ner_result)

        return render_template("stanza.html",
            text = text,
            token_counter = f'{token_counter} tokens in your text',
            sentences_counter = f'{sentences_counter} sentences in your text',
            entities_counter = f'{entities_counter} entities found in your text',
            dataset = [stanza_result_df.to_html(classes= 'data')],
            ent_dataset = [df_entities.to_html(classes= 'data')])

    return render_template("stanza.html")

#####################################################
############### SPACY 
#####################################################
@app.route('/spacy_page', methods = ['GET', 'POST'])
def spacy_page() :

    if request.method == 'POST' :

        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            spacy_nlp = spacy_en
        else :
            spacy_nlp = spacy_fr

        # Return tokenization as Document object
        spacy_doc = spacy_nlp(text)

        # Entity detection graph
        ent_graph = displacy.render(spacy_doc, style="ent")

        return render_template("spacy.html",
            text = text,
            ent_graph = ent_graph)

    return render_template("spacy.html")

#####################################################
############### Text Razor 
#####################################################
@app.route('/razor_page', methods = ['GET', 'POST'])
def razor_page() :

    if request.method == 'POST' :

        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            stanza_nlp = stanza_en
        else :
            stanza_nlp = stanza_en

        # Return tokenization as Document object
        stanza_doc = stanza_nlp(text)

        # Document conversion to dict for manipulation
        stanza_result = stanza_doc.to_dict()

        # Lists preparation for DataFrame
        text_list = []
        lemma_list = []
        upos_list = []

        # We loop inside result dict to append lists
        for sentence in range(len(stanza_result)) :
            for word in range(len(stanza_result[sentence])) :
                text_list.append(stanza_result[sentence][word]['text'])
                lemma_list.append(stanza_result[sentence][word]['lemma'])
                upos_list.append(stanza_result[sentence][word]['upos'])

        # We display result in a dataframe
        stanza_result_df = pd.DataFrame({
            'Text' : text_list,
            'Lemma' : lemma_list,
            'Part Of Speech (pos)' : upos_list})

        # Return entities as JSON object
        stanza_ner_result = stanza_doc.entities
        # Results format
        ent_text_list = []
        ent_type_list = []

        for entity in range(len(stanza_ner_result)) :
            ent_text_list.append(stanza_ner_result[entity].text)
            ent_type_list.append(stanza_ner_result[entity].type)

        df_entities = pd.DataFrame({
            'Text' : ent_text_list,
            'Type' : ent_type_list})

        # Counters
        sentences_counter = len(stanza_doc.sentences)
        token_counter = len(text_list)
        entities_counter = len(stanza_ner_result)

        return render_template("razor.html",
            text = text,
            token_counter = f'{token_counter} tokens in your text',
            sentences_counter = f'{sentences_counter} sentences in your text',
            entities_counter = f'{entities_counter} entities found in your text',
            dataset = [stanza_result_df.to_html(classes= 'data')],
            ent_dataset = [df_entities.to_html(classes= 'data')])

    return render_template("razor.html")

#####################################################
############### Flair 
#####################################################
@app.route('/flair_page', methods = ['GET', 'POST'])
def flair_page() :

    if request.method == 'POST' :

        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            stanza_nlp = stanza_en
        else :
            stanza_nlp = stanza_en

        # Return tokenization as Document object
        stanza_doc = stanza_nlp(text)

        # Document conversion to dict for manipulation
        stanza_result = stanza_doc.to_dict()

        # Lists preparation for DataFrame
        text_list = []
        lemma_list = []
        upos_list = []

        # We loop inside result dict to append lists
        for sentence in range(len(stanza_result)) :
            for word in range(len(stanza_result[sentence])) :
                text_list.append(stanza_result[sentence][word]['text'])
                lemma_list.append(stanza_result[sentence][word]['lemma'])
                upos_list.append(stanza_result[sentence][word]['upos'])

        # We display result in a dataframe
        stanza_result_df = pd.DataFrame({
            'Text' : text_list,
            'Lemma' : lemma_list,
            'Part Of Speech (pos)' : upos_list})

        # Return entities as JSON object
        stanza_ner_result = stanza_doc.entities
        # Results format
        ent_text_list = []
        ent_type_list = []

        for entity in range(len(stanza_ner_result)) :
            ent_text_list.append(stanza_ner_result[entity].text)
            ent_type_list.append(stanza_ner_result[entity].type)

        df_entities = pd.DataFrame({
            'Text' : ent_text_list,
            'Type' : ent_type_list})

        # Counters
        sentences_counter = len(stanza_doc.sentences)
        token_counter = len(text_list)
        entities_counter = len(stanza_ner_result)

        return render_template("flair.html",
            text = text,
            token_counter = f'{token_counter} tokens in your text',
            sentences_counter = f'{sentences_counter} sentences in your text',
            entities_counter = f'{entities_counter} entities found in your text',
            dataset = [stanza_result_df.to_html(classes= 'data')],
            ent_dataset = [df_entities.to_html(classes= 'data')])

    return render_template("flair.html")

#####################################################
############### CAMEMBERT 
#####################################################
@app.route('/camembert_page', methods = ['GET', 'POST'])
def camembert_page() :

    if request.method == 'POST' :

        text = request.form.get('text_input')
        language = request.form.get('language')

        if language == 'en' :
            stanza_nlp = stanza_en
        else :
            stanza_nlp = stanza_en

        # Return tokenization as Document object
        stanza_doc = stanza_nlp(text)

        # Document conversion to dict for manipulation
        stanza_result = stanza_doc.to_dict()

        # Lists preparation for DataFrame
        text_list = []
        lemma_list = []
        upos_list = []

        # We loop inside result dict to append lists
        for sentence in range(len(stanza_result)) :
            for word in range(len(stanza_result[sentence])) :
                text_list.append(stanza_result[sentence][word]['text'])
                lemma_list.append(stanza_result[sentence][word]['lemma'])
                upos_list.append(stanza_result[sentence][word]['upos'])

        # We display result in a dataframe
        stanza_result_df = pd.DataFrame({
            'Text' : text_list,
            'Lemma' : lemma_list,
            'Part Of Speech (pos)' : upos_list})

        # Return entities as JSON object
        stanza_ner_result = stanza_doc.entities
        # Results format
        ent_text_list = []
        ent_type_list = []

        for entity in range(len(stanza_ner_result)) :
            ent_text_list.append(stanza_ner_result[entity].text)
            ent_type_list.append(stanza_ner_result[entity].type)

        df_entities = pd.DataFrame({
            'Text' : ent_text_list,
            'Type' : ent_type_list})

        # Counters
        sentences_counter = len(stanza_doc.sentences)
        token_counter = len(text_list)
        entities_counter = len(stanza_ner_result)

        return render_template("camembert.html",
            text = text,
            token_counter = f'{token_counter} tokens in your text',
            sentences_counter = f'{sentences_counter} sentences in your text',
            entities_counter = f'{entities_counter} entities found in your text',
            dataset = [stanza_result_df.to_html(classes= 'data')],
            ent_dataset = [df_entities.to_html(classes= 'data')])

    return render_template("camembert.html")


if __name__ == "__main__" :
    app.run(debug=True)
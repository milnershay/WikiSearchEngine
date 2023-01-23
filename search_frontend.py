from flask import Flask, request, jsonify
import inverted_index_gcp
import pandas
from pathlib import Path
import pickle
import math
from collections import Counter
import nltk as nltk
import re
import pyspark
import sys
import numpy as np
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
from nltk.corpus import stopwords
nltk.download('stopwords')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
'''
reading the pkl files and building all global variables
'''
maxPV = 181126232
corpus_size = 6348910
title_index = inverted_index_gcp.InvertedIndex.read_index('title2', 'title_index')
body_index = inverted_index_gcp.InvertedIndex.read_index('body', 'index')
anchor_index = inverted_index_gcp.InvertedIndex.read_index('anchor', 'anchor_index')
pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
p = Path(pv_path)
it_dic = {}
dl_dic = {}
pr_dic = {}
pv_clean = f'uploads/{p.stem}.pkl'
with open(pv_clean, 'rb') as f:
    pageView = pickle.loads(f.read())
with open(f'uploads/it_dic.pickle', 'rb') as handle1:
    it_dic = pickle.load(handle1)
with open(f'uploads/dl_dic.pickle', 'rb') as handle2:
    dl_dic = pickle.load(handle2)
with open(f'uploads/pr_dic.pickle', 'rb') as handle3:
    pr_dic = pickle.load(handle3)
global maxPR
maxPR = 9913.72878216077
global avg_dl
global max_dl
max_dl = 387391.0
avg_dl = 3442.288559
page_rank = np.array(list(pr_dic.items()))
page_views = np.array(list(pageView.items()))
page_rank[:,1] = page_rank[:,1]/maxPR
page_views[:,1] = page_views[:,1]/maxPV
page_rank[:,1] = page_rank[:,1]*0.14987132
page_views[:,1] = page_views[:,1]*0.16796444


def map_ids(original_ids):
    id_mapping = {original_id: new_index for new_index, original_id in enumerate(original_ids)}
    return id_mapping


id_map = map_ids(dl_dic.keys())


def group_by_sum(x):  # group np array
    u, idx = np.unique(x[:,0], return_inverse=True)
    s = np.bincount(idx, weights = x[:,1])
    return np.c_[u, s]


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://34.172.191.245:8080/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # get all scores as np arrays
    scores = np.append(page_rank, page_views, axis=0)
    body_scores = calc_bm25_and_cosine_body(query)
    body_scores = body_scores
    anchor_scores = np.array(list(anchor_for_optimization(query)))
    anchor_scores = anchor_scores[(-anchor_scores[:, 1]).argsort()][:200]
    title_scores = np.array(list(title_for_optimization(query)))
    #normalize and add weights to the scores
    anchor_scores[:,1] = anchor_scores[:,1]*(0.16934626/anchor_scores[:,1].max())
    if len(title_scores)>0:
        title_scores = title_scores[(-anchor_scores[:, 1]).argsort()][:200]
        title_scores[:,1] = title_scores[:,1]*0.09949586/(title_scores[:,1].max())
        scores = np.append(scores, title_scores, axis=0)

    # append all scores to one np array
    scores = np.append(scores, body_scores, axis=0)
    scores = np.append(scores, anchor_scores, axis=0)
    # scores group by id
    scores = group_by_sum(scores)
    # sort in descending order
    idx = (-scores[:,1]).argsort()
    scores = scores[idx]
    # get top 100 pages
    scores = scores[:100]

    best_ids = scores[:,0]
    for p_id in best_ids:
        p_id = int(p_id)
        try:
            title = it_dic[p_id]
            res.append((p_id, title))
        except:
            print("couldn't find title")
            continue

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://35.193.30.223:8080/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    scores = calc_body_cosine(query)
    best = scores.most_common(100)
    for p_id, score in best:
        try:
            title = it_dic[p_id]
            res.append((p_id, title))
        except:
            continue

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title(query=None):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    count = []
    q = tokenize(query)
    q = list(set(q))
    for t in q:
        pl = get_posting_lists(title_index, [t], 'title2')
        cur = []
        for p_id, value in pl:
            cur.append(p_id)
        cur = list(set(cur))
        count += cur
    scores = Counter(count)
    scores = scores.most_common()
    for p_id, value in scores:
        try:
            title = it_dic[p_id]
            res.append((p_id, title))
        except:
            continue
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor(query=None):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    q = tokenize(query)
    pl = get_posting_lists(anchor_index, q, 'anchor')
    for p_id, value in pl:
        try:
            title = it_dic[p_id]
            res.append((p_id, title))
        except:
            continue

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank(wiki_ids=None):
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    if wiki_ids is None:
        wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        try:
            rank = pr_dic[doc_id]
            res.append(rank)
        except:
            continue
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview(wiki_ids=None):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    if wiki_ids is None:
        wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        try:
            res.append(pageView[doc_id])
        except:
            continue
    # END SOLUTION
    return jsonify(res)


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(index, w, base_dir=''):
    '''
    method to read posting lists from disk (bin files)
    gets inverted index (index) word (w) and directory
    returns the posting_lists for word w in the inverted index
    '''
    with closing(inverted_index_gcp.MultiFileReader()) as reader:
        try:
            locs = index.posting_locs[w]
            new_locs = [tuple((base_dir + '/' + locs[0][0], locs[0][1]))]
            b = reader.read(new_locs, index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list
        except:
            return []


def get_posting_lists(index, query, base_dir=''):
    '''
    method to get the posting lists relevant to the query
    gets inverted index, query, and directory
    returns a posting list
    '''
    posting_lists = []
    for w in query:
        posting_list = read_posting_list(index, w, base_dir)
        posting_lists += posting_list
    pl = Counter()
    for pair in posting_lists:
        pl[pair[0]] += pair[1]
    return pl.most_common()


def tokenize(query):
    '''
    a method to tokenize query
    gets query string
    returns list of tokens
    '''
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [token for token in tokens if token not in all_stopwords]
    return tokens


def calc_bm25_and_cosine_body(query):
    '''
    a function to calaulate bm25 scores on body
    '''
    b = 0.75
    k1 = 1.5
    q = tokenize(query)
    q_wc = Counter(q)
    bm25 = []
    cosine = []
    # scores = Counter()
    for t in q:
        pl = get_posting_lists(body_index, [t], 'body')
        idf1 = math.log10(((corpus_size - body_index.df[t] + 0.5) / (body_index.df[t] + 0.5)) + 1)
        idf2 = math.log10(corpus_size / body_index.df[t])
        for p_id, tf in pl:
            dl = dl_dic.get(p_id, avg_dl)
            bm25.append((p_id, idf1 * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl / avg_dl))))))
            tfidf = tf * idf2
            cosine.append((p_id, tfidf * q_wc[t]))
    bm25 = np.array(bm25)
    bm25 = group_by_sum(bm25)
    bm25[:,1] = bm25[:,1] * (0.23449732 /(bm25[:,1].max()))
    cosine = np.array(cosine)
    cosine = group_by_sum(cosine)
    cosine[:,1] = cosine[:,1] * (0.17882479 /(cosine[:,1].max()))
    scores = np.append(cosine,bm25,axis=0)
    scores = group_by_sum(scores)
    return scores[(-scores[:, 1]).argsort()][:200]


def calc_body_cosine(query):
    '''
    calculates cosine sim scores for query
    '''
    q = tokenize(query)
    q_wc = Counter(q)
    scores = Counter()
    for t in q:
        pl = get_posting_lists(body_index, [t], 'body')
        idf = math.log10(corpus_size / body_index.df[t])
        for p_id, tf in pl:
            tfidf = tf * idf
            scores[p_id] += tfidf * q_wc[t]

    return scores


@app.route("/bm25")
def bm25_for_optimization():
    '''
    a function that returns a set of bm25 scores per query for optimization purposes
    '''
    query = request.args.get('query', '')
    b = 0.75
    k1 = 1.5
    q = tokenize(query)
    scores = Counter()
    for t in q:
        pl = get_posting_lists(body_index, [t], 'body')
        idf = math.log10(((corpus_size - body_index.df[t] + 0.5)/(body_index.df[t] + 0.5)) + 1)
        for p_id, tf in pl:
            dl = dl_dic.get(p_id, avg_dl)
            scores[p_id] += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl / avg_dl))))

    return jsonify(scores)


@app.route("/tfidf")
def tfidf_for_optimization():
    '''
    a function that returns a set of cosine sim scores per query for optimization purposes
    '''
    query = request.args.get('query', '')
    q = tokenize(query)
    q_wc = Counter(q)
    scores = Counter()
    for t in q:
        pl = get_posting_lists(body_index, [t], 'body')
        idf = math.log10(corpus_size / body_index.df[t])
        for p_id, tf in pl:
            tfidf = tf * idf
            scores[p_id] += tfidf * q_wc[t]

    return jsonify(scores)


@app.route("/title")
def title_for_optimization(query=None):
    '''
    a function that returns a set of title scores per query for optimization purposes
    '''
    flag = True
    if query is None:
        query = request.args.get('query', '')
        flag = False
    q = tokenize(query)
    title_scores = get_posting_lists(title_index, q, 'title2')

    if flag:
        return title_scores
    return jsonify(title_scores)


@app.route("/anchor")
def anchor_for_optimization(query):
    '''
    a function that returns a set of anchor scores per query for optimization purposes
    '''
    flag = True
    if query is None:
        query = request.args.get('query', '')
        flag = False
    q = tokenize(query)
    anchor_scores = get_posting_lists(anchor_index, q, 'anchor')

    if flag:
        return anchor_scores

    return jsonify(anchor_scores)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

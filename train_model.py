from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import requests
import time as t
from time import time
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
p = Path(pv_path)
pr_dic = {}
dl_dic = {}
num_doc = 6348910
the_missing_keys = []
url = 'http://34.28.8.128:8080'
pv_clean = f'uploads/{p.stem}.pkl'
with open(pv_clean, 'rb') as f:
    pageView = pickle.loads(f.read())
with open(f'uploads/it_dic.pickle', 'rb') as handle1:
    it_dic = pickle.load(handle1)
with open(f'uploads/dl_dic.pickle', 'rb') as handle2:
    dl_dic = pickle.load(handle2)
with open(f'uploads/pr_dic.pickle', 'rb') as handle3:
    pr_dic = pickle.load(handle3)

id_map = {}
id_len = 0

def map_ids(original_ids):
    id_mapping = {original_id: new_index for new_index, original_id in enumerate(original_ids)}
    return id_mapping


def make_keymapper():
    print("=========== MAKING KEY MAPPER ============")
    keys = []
    keys += list(pr_dic.keys())
    keys += list(dl_dic.keys())
    keys += list(it_dic.keys())
    keys += list(pageView.keys())

    keys = list(set(keys))
    id_map = map_ids(keys)
    id_len = len(id_map)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


# place the domain you got from ngrok or GCP IP below.
# url = 'http://XXXX-XX-XX-XX-XX.ngrok.io'


def run_queries(queries):
    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        try:
            res = requests.get(url + '/search?', {'query': q}, timeout=35)
            duration = time() - t_start
            pred_wids, _ = zip(*res.json())
            ap = average_precision(true_wids, pred_wids)
        except:
            pass
        if ap is None:
            ap = 0
        if duration is None:
            duration = 0
        qs_res.append((q, duration, ap))
        print(q + '\t' + str(ap))

    mean_duration = sum(elt[1] for elt in qs_res) / len(qs_res)
    mean_ap = sum(elt[2] for elt in qs_res) / len(qs_res)
    return mean_duration, mean_ap


# Prepare the training data
def prepare_train_data(queries):
    data = []
    for q, docs in queries.items():
        ans = [0] * id_len
        for doc in docs:
            ans[id_map[doc]] = 1
        data.append((q, ans))
    return data


def get_all_scores(queries):
    tfidf = {}
    bm25 = {}
    anchor = {}
    title = {}
    count = 29
    pr = [0] * id_len
    pv = [0] * id_len
  #  for k in pr_dic.keys():
  #      if int(k) in id_map.keys():
  #          pr[id_map[int(k)]] = pr_dic[k] / 9913.72878216077
  #      else:
  #          the_missing_keys.append(k)
  #  print("=========== CALCULATED PAGERANK ================")
  #  for k in pageView.keys():
  #      if k in id_map.keys():
  #          pv[id_map[k]] = pageView[k] / 181126232
  #  print("=========== CALCULATED PAGEVIEW ================")
    for q in queries.keys():
     #   tf = requests.get(url + '/tfidf?', {'query': q})
     #   tf_scores = tf.json()
     #   tf_true = [0] * id_len
     #   for k in tf_scores.keys():
     #       if int(k) in id_map.keys():
     #           tf_true[id_map[int(k)]] = float(tf_scores[k])
     #       else:
     #           the_missing_keys.append(k)
     #   tfidf[q] = tf_true
        bm = requests.get(url + '/bm25?', {'query': q})
        print("done reading")
        bm_score = bm.json()
        bm_true = [0] * id_len
        for k in bm_score.keys():
            if int(k) in id_map.keys():
                bm_true[id_map[int(k)]] = float(bm_score[k])
            else:
                the_missing_keys.append(k)
        bm25[q] = bm_true
        anc = requests.get(url + '/anchor?', {'query': q})
        anc_scores = anc.json()
        anc_true = [0] * id_len
        for k, v in anc_scores:
            if int(k) in id_map.keys():
                anc_true[id_map[int(k)]] = float(v)
            else:
                the_missing_keys.append(k)
        anchor[q] = anc_true
        tit = requests.get(url + '/title?', {'query': q})
        title_scores = tit.json()
        title_true = [0] * id_len
        for k, v in title_scores:
            if int(k) in id_map.keys():
                title_true[id_map[int(k)]] = v
            else:
                the_missing_keys.append(k)
        title[q] = title_true
        print("=========== CALCULATED QUERY " + q + " - " + str(count) + " left   ================")
        count -= 1

    return tfidf, bm25, anchor, title, pr, pv


# train the model and find best weights with gridSearch and linearReg
def train_my_boy(tfidf_scores, bm25_scores, anchor_scores, title_scores, pagerank_scores, pageview_scores,
                 training_data):
    # prepare the input and target data
    y = []
    X = []
    tf_vec = []
    bm_vec = []
    anc_vec = []
    tit_vec = []
    pr_vec = []
    pv_vec = []
    t_vec = []
    print("Final processing for all data: " +  t.strftime("%H:%M:%S", t.localtime()))
    for query, relevant_docs in training_data:
        tf_vec += tfidf_scores[query]
        bm_vec += bm25_scores[query]
        anc_vec += anchor_scores[query]
        tit_vec += title_scores[query]
        pr_vec += pagerank_scores
        pv_vec += pageview_scores
        t_vec += relevant_docs
    print("made X and y, transposing time! : " + t.strftime("%H:%M:%S", t.localtime()))
    X = [tf_vec, bm_vec, anc_vec, tit_vec, pr_vec, pv_vec]
    y = t_vec
    X = np.array(X).transpose()
    print("Done all data: " + t.strftime("%H:%M:%S", t.localtime()))

   # print("X and y is almost ready, needs reshape, time: " + t.strftime("%H:%M:%S", t.localtime()))

   # print("Done reshaping! : " +t.strftime("%H:%M:%S", t.localtime()))


    # Create the Linear Regression model
    lin_reg = LinearRegression()
    # Fit the model to the data
    print("Fitting with linear regression: started at " + t.strftime("%H:%M:%S", t.localtime()))
    lin_reg.fit(X, y)
    print("Done linear regression: " +  t.strftime("%H:%M:%S", t.localtime()))
    # Get the feature weights
    weights = lin_reg.coef_

    print("Feature weights: ", weights)

    print("Creating model with random forest")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    print("Fitting with random forest and full data: started at " + t.strftime("%H:%M:%S", t.localtime()))
    clf.fit(X, y)
    print("Done random forest full: " +  t.strftime("%H:%M:%S", t.localtime()))
    weights = clf.feature_importances_
    print("Optimized weights:", weights)


def check_model():
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)
    print("=============TESTING WITH JSON 1===============")
    mean_duration, mean_ap = run_queries(queries)
    print("=============AVG DURATION - JSON 1===============")
    print(mean_duration)
    print("=============AVG AP SCORE - JSON 1===============")
    print(mean_ap)

    with open('new_train.json', 'rt') as f:
        queries = json.load(f)
    print("=============TESTING WITH JSON 2===============")
    mean_duration, mean_ap = run_queries(queries)
    print("=============AVG DURATION - JSON 2===============")
    print(mean_duration)
    print("=============AVG AP SCORE - JSON 2===============")
    print(mean_ap)


def pickle_Saver(queries):
    print("=========== ASKING QUERIES: " +t.strftime("%H:%M:%S", t.localtime()) + " ============")
    tfidf, bm25, anchor, title, pr, pv = get_all_scores(queries)
    print("=========== CREATING PICKLES: " +t.strftime("%H:%M:%S", t.localtime()) + "  ============")
    with open('optimization/tf_scores.pkl', 'wb') as pik1:
        pickle.dump(tfidf, pik1)
    with open('optimization/bm_scores.pkl', 'wb') as pik2:
        pickle.dump(bm25, pik2)
    with open('optimization/anc_scores.pkl', 'wb') as pik3:
        pickle.dump(anchor, pik3)
    with open('optimization/title_scores.pkl', 'wb') as pik4:
        pickle.dump(title, pik4)


def main():
    check_model()
#    with open('new_train.json', 'rt') as f:
#        queries = json.load(f)
#    print("=========== MAKING DATA: " +t.strftime("%H:%M:%S", t.localtime()) + " ============")
#    data = prepare_train_data(queries)
#    print("=========== OPENING PICKLES: " + t.strftime("%H:%M:%S", t.localtime()) + "  ============")
#    with open('optimization/tf_scores.pkl', 'rb') as pik1:
#        tfidf = pickle.load(pik1)
#    with open('optimization/bm_scores.pkl', 'rb') as pik2:
#        bm25 = pickle.load(pik2)
#    with open('optimization/anc_scores.pkl', 'rb') as pik3:
#        anchor = pickle.load(pik3)
#    with open('optimization/title_scores.pkl', 'rb') as pik4:
#        title = pickle.load(pik4)
#    with open('optimization/pr_scores.pkl', 'rb') as pik5:
#        pr = pickle.load(pik5)
#    with open('optimization/pv_scores.pkl', 'rb') as pik6:
#        pv = pickle.load(pik6)

#    print("============ TRAINING TIME: " + t.strftime("%H:%M:%S", t.localtime()) + "  =============")
#    train_my_boy(tfidf, bm25, anchor, title, pr, pv, data)





if __name__ == '__main__':
    main()

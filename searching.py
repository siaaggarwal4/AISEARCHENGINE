import json
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import os
import re
import warnings
from nltk.stem import PorterStemmer
from urllib.parse import urljoin, urldefrag
import math
import time
import numpy as np

TITLEEMBEDDINGSPATH = "titleembed.json"

# from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

stemmer = PorterStemmer()

INDEX_PATH    = "index.json"
URLMAP_PATH   = "url_map.json"
SEEK_PATH     = "seek_index.json"
CHAMPION_PATH = "champion_lists.json"
ANCHOR_PATH   = "anchor_index.json"
EMBEDDING_PATH = "embeddings.json"

EXACT_HASH_PATH = "exact_hashes.json"
SIMHASH_PATH    = "simhashes.json"

ALPHA         = 0.7
ANCHOR_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.5
# TITLEEMBED_WEIGHT = 0.2
model = SentenceTransformer('all-MiniLM-L6-v2')

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def s_hash(text):
    hash_val = 5381
    for c in text:
        hash_val = ((hash_val << 5) + hash_val) + ord(c)

    hash_val = abs(hash_val) % (2 ** 64)

    return format(hash_val, f'64b')


def hd(x1, x2):
    c = 0
    x1 = str(x1)
    for i in range(len(x1)):
        if x1[i] == x2[i]:
            continue
        else:
            c += 1
    return c


def load_index(index_path, urlmap_path):
    with open(urlmap_path, "r") as f:
        url_map = json.load(f)
    with open(SEEK_PATH, "r") as f:
        seek = json.load(f)
    with open(CHAMPION_PATH, "r") as f:
        champions = json.load(f)
    with open(EMBEDDING_PATH) as f:
        saved_doc = json.load(f)
        # saved_doc = np.load(EMBEDDING_PATH)
    with open(TITLEEMBEDDINGSPATH) as file:
        titlembed = json.load(file)


    anch_index = {}
    if os.path.exists(ANCHOR_PATH):
        with open(ANCHOR_PATH, "r") as f:
            anch_index = json.load(f)

    exact_hashes = {}
    if os.path.exists(EXACT_HASH_PATH):
        with open(EXACT_HASH_PATH, "r") as f:
            exact_hashes = json.load(f)

    simhashes = {}
    if os.path.exists(SIMHASH_PATH):
        with open(SIMHASH_PATH, "r") as f:
            simhashes = json.load(f)

    return seek, url_map, champions, saved_doc,titlembed, anch_index, exact_hashes, simhashes # pr_scores,


def get_postings_from_disk(term, seek):
    if term not in seek:
        return {}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        f.seek(seek[term])
        line = f.readline()
    try:
        return json.loads(line)[term]
    except Exception:
        return {}


def stem_query(query_text):
    # return list of stemm word
    tokens = tokenize(query_text)
    return [stemmer.stem(t) for t in tokens]


def boolean_and_retrieve(query_terms, seek):

    if not query_terms:
        return set()

    posting_sets = []
    for term in query_terms:
        if term in seek:
            posting_sets.append(set(get_postings_from_disk(term, seek).keys()))
        else:
            return set()

    result = posting_sets[0]
    for ps in posting_sets[1:]:
        result = result.intersection(ps)
    
    if len(result) < 10:
        return posids_retrieve(query_terms, seek) # we run the query parser

    return result


def position_retrival(query_terms, seek):
    word = query_terms[0]
    dw1 = get_postings_from_disk(word, seek)
    if not dw1:
        return {}

    for w in query_terms[1:]:
        wds = get_postings_from_disk(w, seek)
        if not wds:
            return {}
        next_result = {}
        for elem in dw1:
            if elem in wds:
                if len(dw1[elem]) < 2 or len(wds[elem]) < 2:
                    continue
                l1 = dw1[elem][1] if isinstance(dw1[elem][1] , list) else [dw1[elem][1] ]
                l2 = wds[elem][1] if isinstance(wds[elem][1], list ) else [wds[elem][1] ]
                i1 = 0
                i2 = 0
                s = []
                while i1 < len(l1) and i2 < len(l2):
                    if l2[i2] == l1[i1]+1:
                        s.append(l2[i2])
                        i1 += 1
                        i2 += 1
                    elif l2[i2] < l1[i1]+1:
                        i2 += 1
                    else:
                        i1 += 1
                if s:
                    next_result[elem] = s
        dw1 = next_result

    return dw1


def posids_retrieve(query_terms, seek):
    if not query_terms:
        return set()

    result = {}
    result.update(position_retrival(query_terms, seek))

    for i in range(2, len(query_terms)+1):
        if len(result) > 12:
            break
        for j in range(len(query_terms) - i + 1):
            sub = query_terms[j:j+i]
            result.update(position_retrival(sub, seek))

    return set(result.keys())


def bi_gram_search(query_terms, seek):
    posting_sets = {}
    common = {"to", "the"}
    common_bigrams = {("to", "the")}
    for w in range(len(query_terms)-1):
        w1 = query_terms[w]
        w2 = query_terms[w+1]
        if w1 in common or w2 in common or (w1, w2) in common_bigrams:
            break
        w1l = get_postings_from_disk(w1, seek)
        w2l = get_postings_from_disk(w2, seek)

        if not w1l or not w2l:
            continue

        for elem in w1l:
            if elem in w2l:
                l1 = w1l[elem][1]
                l2 = w2l[elem][1]

                i1 = 0
                i2 = 0

                while i1 < len(l1) and i2 < len(l2):
                    if l2[i2] == l1[i1]+1:
                        if elem in posting_sets:
                            posting_sets[elem] += 1
                        else:
                            posting_sets[elem] = 1
                        i1 += 1
                        i2 += 1
                    elif l2[i2] < l1[i1]+1:
                        i2 += 1
                    else:
                        i1 += 1

    return posting_sets


def rank_documents(model, choosedoc, query_terms, seek, total_docs, save_doc, anchor_index=None): # pagerank_scores=None, 
    scores = {}

    # print()
    for term in query_terms:
        postings = get_postings_from_disk(term, seek)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(total_docs / df) if df > 0 else 0

        for doc in choosedoc:
            doc_str = str(doc) if isinstance(doc, int) else doc
            if doc_str in postings:
                tf_raw = postings[doc_str][0]
                tf = 1 + math.log(tf_raw) if tf_raw > 0 else 0
                scores[doc_str] = scores.get(doc_str, 0) + (tf * idf)

    print("computing tfidf for query")

    if not scores:
        return []

    max_tfidf  = max(scores.values())
    norm_tfidf = {d: s / max_tfidf for d, s in scores.items()} if max_tfidf > 0 else scores

    print(" calculating anchor indexes")

    anchor_scores = {}
    if anchor_index:
        for term in query_terms:
            if term not in anchor_index:
                continue
            for doc_str, count in anchor_index[term].items():
                if doc_str in scores:
                    anchor_scores[doc_str] = anchor_scores.get(doc_str, 0) + count
        max_anchor = max(anchor_scores.values()) if anchor_scores else 1.0
        if max_anchor > 0:
            anchor_scores = {d: s / max_anchor for d, s in anchor_scores.items()}

    print("calculating semantic scores")

    semantic_scores = {}

    query_emb = model.encode_query(query_terms, normalize_embeddings=True)
    
    for doc in choosedoc: # get each doc's embedding
        doc_embedding = save_doc[int(doc)]["chunk_embeddings"]
        scoresd = set()
        for chunk_embedding in doc_embedding:
            # print(np.dot(query_emb, chunk_embedding)[0])
            scoresd.add(np.dot(query_emb, np.array(chunk_embedding))[0])
        semantic_scores[doc] = max(max(scoresd), 0)
    

    combined = {}
    for doc_str in scores:
        tfidf  = norm_tfidf.get(doc_str, 0.0)
        anchor = anchor_scores.get(doc_str, 0.0)
        semantic = semantic_scores.get(doc_str, 0.0)
        combined[doc_str] = ALPHA * tfidf  + ANCHOR_WEIGHT * anchor+ SEMANTIC_WEIGHT *semantic # (1 - ALPHA) * pr

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked


def exact_check(positing_lists, exact_hashes):
    # remove exact docs
    positing_list = list(positing_lists)
    seen = {}
    i = 0
    while i < len(positing_list):
        h = exact_hashes.get(positing_list[i], None)
        if h is None:
            i += 1
            continue
        if h in seen:
            positing_list.pop(i)
        else:
            seen[h] = True
            i += 1
    return positing_list


def near_check(positings_list, simhashes):
    near = set()
    positings_list = list(positings_list)
    i = 0
    while i < len(positings_list):
        h1 = simhashes.get(positings_list[i], None)
        if h1 is None:
            i += 1
            continue
        j = i + 1
        while j < len(positings_list):
            h2 = simhashes.get(positings_list[j], None)
            if h2 is None:
                j += 1
                continue
            dist = hd(h1, h2)
            if dist < 3:
                # near[positings_list[j]] = True
                near.add(positings_list[j])
                positings_list.pop(j)
            else:
                j += 1
        i += 1
    print(near)
    return positings_list, near

def run_titleembedding_similarity(embed, query, urlmap):
    s = -1
    slist = []
    query_embedding = model.encode_query(query)
    # print(query_embedding.shape)
    # i=0
    for embedding in embed:
        embeding = np.array(embedding["chembedding"])
        # print(embeding.shape)
        try:
            cosinesim = np.dot(embeding, query_embedding)
        except Exception:
            pass
        # cosinesim=1

        slist.append([embedding["doc_id"], cosinesim])
        # i+=1
    # print(slist)
    slist.sort(key=lambda x:x[1], reverse=True)
    # print(slist[:10])
    for i in range(len(slist)):
        url = urlmap[str(slist[i][0])]
        slist[i][0] = url
        slist[i][1] *= 100

    return slist[:10]
                     
def search(query_text, seek, url_map, champions,
           saved_docs,title_embed, anchor_index=None,
           exact_hashes=None, simhashes=None, top_k=5): # pagerank_scores=None, 
    start = time.time()
    print("starting search now..")

    query_terms = stem_query(query_text)
    if not query_terms:
        
        return [], 0, []

    total_docs = len(url_map)

    print("running boolean")
    candidate_docs = boolean_and_retrieve(query_terms, seek)
    print(len(candidate_docs))

    if not candidate_docs:
        # run ai semantic search on titles indexes
        
        docs = run_titleembedding_similarity(title_embed, query_text, urlmap=url_map)
        elapsed = (time.time() - start) * 1000
        if docs:
            return docs, elapsed, []
        return [], elapsed, []

    champion_docs = set() 
    for term in query_terms:
        if term in champions:
            champion_docs.update(champions[term])
    # print(champion_docs)

    filtered_docs = candidate_docs.intersection(champion_docs)

    if not filtered_docs:
        filtered_docs = candidate_docs
    
    print("filtered docs", filtered_docs)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # print(model)
    print("going to rank")
    # saved
    ranked = rank_documents(model, filtered_docs, query_terms, seek, total_docs, saved_docs,
                             anchor_index=anchor_index ) # pagerank_scores=pagerank_scores, 

    bigram_hits = bi_gram_search(query_terms, seek)
    positional_hits = posids_retrieve(query_terms, seek)
    if bigram_hits or positional_hits:
        boosted = []
        for doc_str, score in ranked:
            bonus = bigram_hits.get(doc_str, 0) * 0.1
            if doc_str in positional_hits:
                bonus += 0.15
            boosted.append((doc_str, score + bonus))
        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)

    top_pool = [doc_str for doc_str, _ in ranked[:top_k * 3]]
    nearsim=[]
    if exact_hashes:
        top_pool = exact_check(top_pool, exact_hashes)
    if simhashes:
        top_pool, nearsim = near_check(top_pool, simhashes)
    # near
    results = []
    seen_urls = set()
    score_map = {doc_str: score for doc_str, score in ranked}
    for doc_str in top_pool:
        url = url_map.get(doc_str, url_map.get(int(doc_str), "UNKNOWN"))
        if url not in seen_urls:
            seen_urls.add(url)
            results.append((url, score_map.get(doc_str, 0)))
        if len(results) == top_k:
            break
    
    nearsimdocs = []
    for doc_str in nearsim:
        nearsimdocs.append(url_map.get((int(doc_str), "unknown")))
    
    elapsed = (time.time() - start) * 1000

    return results, elapsed, nearsimdocs

def search_cli(query=None, cli=False):
    print("\n===== LOADING INDEX FOR SEARCH =====")
    seek, url_map, champions, saved_doc, title_embeddings, anch_index, exact_hashes, simhashes = load_index(INDEX_PATH, URLMAP_PATH) # pr_scores,
    print(f"Index loaded: {len(seek)} tokens, {len(url_map)} documents")
    # print(f"PageRank loaded: {len(pr_scores)} pages")
    print(f"Anchor index loaded: {len(anch_index)} anchor terms")
    print("====================================\n")

    if cli==False:
        results, elapsed_ms, neardocs = search(query, seek, url_map, champions,
                                     saved_doc,title_embeddings, anchor_index=anch_index,
                                     exact_hashes=exact_hashes, simhashes=simhashes, top_k=10)
        
        return results, elapsed_ms, neardocs


    while True:
        query = input("Enter the search query (or enter 'qu' to exit the program): ").strip()
        if query.lower() == "qu":
            print("Thank you")
            break
        if not query:
            continue
        
        print("query to search",  query)
        results, elapsed_ms, neardocs = search(query, seek, url_map, champions,
                                     saved_doc,title_embeddings, anchor_index=anch_index,
                                     exact_hashes=exact_hashes, simhashes=simhashes, top_k=10)

        print(f"\nResults for: \"{query}\"  ({elapsed_ms:.1f} ms)")
        print("-" * 60)
        if results:
            for i, (url, score) in enumerate(results, 1):
                print(f"  {i}. {url}")
                print(f"      Score: {score:.4f}")
        if neardocs:
            print("similar docs:", neardocs)
        else:
            print("  No results found.")
        print("-" * 60)
        print()

# def search_func()


if __name__ == "__main__":
    search_cli("", True)
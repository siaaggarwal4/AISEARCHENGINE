import json
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import os
import re
import warnings
from nltk.stem import PorterStemmer
from urllib.parse import urljoin, urldefrag
import math
import time
from sentence_transformers import SentenceTransformer
from semantic_search import save_embedding
import torch
import numpy as np

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

stemmer = PorterStemmer()

# if stronger then h1 h2 h3 imp

model = SentenceTransformer('all-MiniLM-L6-v2')

class Dele:
    def __init__(self, doc, freq=1, next=None):
        self.doc = doc
        self.freq = freq
        self.next = next
        self.pos = []

    def freq_inc(self):
        self.freq += 1

    def pos_inc(self, positionind):
        self.pos.append(positionind)


class PostingList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.c = 0

    def doc_inll(self, doc, pos):
        # empty list
        if self.head is None:
            node = Dele(doc, 1)
            node.pos_inc(pos)
            self.head = self.tail = node
            self.c += 1
            return

        if self.tail.doc == doc:
            self.tail.freq_inc()
            self.tail.pos_inc(pos)
            return

        # append new doc
        node = Dele(doc, 1)
        node.pos_inc(pos)
        self.tail.next = node
        self.tail = node
        self.c += 1


def get_url_content(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as gh:
        raw = gh.read()
        jo = json.loads(raw)
        return jo["url"], jo["content"]


def content_ht(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    return soup.get_text(separator=" ")


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def get_all_json_files(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))
    return all_files

 

DATA_DIR = "DEV"
FLUSH_LIMIT = 10000
PARTIAL_DIR = "partials"
os.makedirs(PARTIAL_DIR, exist_ok=True)

INDEX_PATH    = "index.json"
URLMAP_PATH   = "url_map.json"
SEEK_PATH     = "seek_index.json"
CHAMPION_PATH = "champion_lists.json"

ANCHOR_PATH   = "anchor_index.json"

EXACT_HASH_PATH = "exact_hashes.json"
SIMHASH_PATH    = "simhashes.json"

EMEDDINGS_PATH = "embeddings.json"
TITLEEMBEDDINGSPATH = "titleembed.json"
ALPHA         = 0.7
ANCHOR_WEIGHT = 0.3
all_files = get_all_json_files(DATA_DIR)

hashtable = {}   # here we map token -> PostingList
url_nums  = {}   # here we map doc_id -> url
url_files = {}   # here we map urls to the files
raw_links = {}   # stores outbound links per doc before resolving to doc ids

doc_id = 0
flush_count = 0
partial_paths = []


# stores {term: {doc_id: [freq, [positions]]}}
def serialize_index(ht):
    serialized = {}
    for token, pl in ht.items():
        postings = {}
        node = pl.head
        while node is not None:
            postings[str(node.doc)] = [node.freq, node.pos]
            node = node.next
        serialized[token] = postings
    return serialized


def flush_partial(ht, fnum):
    path = os.path.join(PARTIAL_DIR, f"partial_{fnum}.json")
    with open(path, "w") as f:
        json.dump(serialize_index(ht), f)
    print(f"flushed partial {fnum} ({len(ht)} terms)")
    return path


def merge_partials(paths):
    merged = {}
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for term, postings in data.items():
            if term not in merged:
                merged[term] = {}
            for doc, val in postings.items():
                if doc in merged[term]:
                    merged[term][doc][0] += val[0]
                    merged[term][doc][1] += val[1]
                else:
                    merged[term][doc] = [val[0], val[1]]
    return merged


# writes one term per line and save byte offset so search can seek() directly to any term
def write_index_lines(merged, out_path):
    seek_map = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for term in sorted(merged.keys()):
            seek_map[term] = f.tell()
            f.write(json.dumps({term: merged[term]}) + "\n")
    return seek_map


def build_champion_lists(index, k=50):
    champions = {}
    for term, postings in index.items():
        sorted_docs = sorted(postings.items(), key=lambda x: x[1][0], reverse=True)
        champions[term] = [doc for doc, _ in sorted_docs[:k]]
    return champions
 

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


def sihash(text, bits=64):
    words = text
    words_freq = {i.lower(): (text.count(i.lower()) + text.count(i)) for i in words}
    f = [0]*64
    for w in words_freq:
        h = s_hash(w)
        s = str(h)
        for i in range(64):
            if s[i] == '1':
                f[i] += words_freq[w]
                continue
            f[i] += -words_freq[w]

    ff = []
    for i in f:
        if i > 0:
            ff.append(1)
        else:
            ff.append(0)

    return str(ff)

saved_doc = []
titles_embedding = []

for file_path in all_files:
    if not file_path.lower().endswith(".json"):
        continue

    try:
        url, raw_content = get_url_content(file_path)
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        continue

    if not raw_content:
        continue

    url = url.split("#")[0]
    url_nums[doc_id] = url
    url_files[url] = file_path
    soup = BeautifulSoup(raw_content, "html.parser")
 
    text = content_ht(raw_content)
    docembedding = save_embedding(model, soup)
    if isinstance(docembedding, np.ndarray): # isinstance(docembedding, np.array):
        saved_doc.append({"doc_id": doc_id, "chunk_embeddings": docembedding.tolist()})
    else:
        saved_doc.append({"doc_id": doc_id, "chunk_embeddings": []})
    
    embed = save_embedding(model, soup, True)
    if isinstance(embed, np.ndarray):
        titles_embedding.append({"doc_id": doc_id, "chembedding": embed.tolist()})
    else:
        titles_embedding.append({"doc_id": doc_id, "chembedding": []})
    
    # embedding = save_embedding(model, soup)
    # embedding = np.asarray(embedding, dtype="float32")
    # saved_doc.append(embedding)

    tokens = tokenize(text)

    for pos, word in enumerate(tokens):
        stem = stemmer.stem(word)

        if stem not in hashtable:
            hashtable[stem] = PostingList()

        hashtable[stem].doc_inll(doc_id, pos)

    doc_raw_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        try:
            full_url, _ = urldefrag(urljoin(url, href))
        except Exception:
            continue
        anchor_text   = a_tag.get_text(strip=True)
        anchor_tokens = [stemmer.stem(t) for t in tokenize(anchor_text)]
        if anchor_tokens:
            doc_raw_links.append((full_url, anchor_tokens))
    raw_links[doc_id] = doc_raw_links

    doc_id += 1

    if doc_id % FLUSH_LIMIT == 0:
        flush_count += 1
        partial_paths.append(flush_partial(hashtable, flush_count))
        hashtable = {}


if hashtable:
    flush_count += 1
    partial_paths.append(flush_partial(hashtable, flush_count))
    hashtable = {}

print(f"total flushes: {flush_count}")

merged_index = merge_partials(partial_paths)
seek_map = write_index_lines(merged_index, INDEX_PATH)

with open(EMEDDINGS_PATH, "w") as f:
    json.dump(saved_doc, f)
# saved_doc = np.asarray(saved_doc)
# np.save(EMEDDINGS_PATH, saved_doc)

with open(TITLEEMBEDDINGSPATH, "w") as f:
    json.dump(titles_embedding, f)

with open(SEEK_PATH, "w") as f:
    json.dump(seek_map, f)

champion_lists = build_champion_lists(merged_index, k=50)
with open(CHAMPION_PATH, "w") as f:
    json.dump(champion_lists, f)

url_to_docid = {u: str(d) for d, u in url_nums.items()}
link_graph   = {}
anchor_index = {}

for src_id, links in raw_links.items():
    src_str = str(src_id)
    targets = []
    for target_url, anchor_tokens in links:
        tgt_str = url_to_docid.get(target_url)
        if tgt_str is None:
            continue
        targets.append(tgt_str)
        for token in anchor_tokens:
            if token not in anchor_index:
                anchor_index[token] = {}
            anchor_index[token][tgt_str] = anchor_index[token].get(tgt_str, 0) + 1
    link_graph[src_str] = list(set(targets))

print(f"link graph: {len(link_graph)} nodes, {sum(len(v) for v in link_graph.values())} edges")
print(f"anchor index: {len(anchor_index)} tokens")


with open(ANCHOR_PATH, "w") as f:
    json.dump(anchor_index, f)

with open(URLMAP_PATH, "w") as f:
    json.dump(url_nums, f)

print("URL mapping saved.")
print("Final index saved.")

print("computing doc hashes...")
exact_hashes = {}
simhashes    = {}
for did, url in url_nums.items():
    fp = url_files.get(url)
    if not fp:
        continue
    try:
        content = get_url_content(fp)[1]
        exact_hashes[str(did)] = s_hash(content)
        simhashes[str(did)]    = sihash(tokenize(content))
    except Exception:
        pass

with open(EXACT_HASH_PATH, "w") as f:
    json.dump(exact_hashes, f)
with open(SIMHASH_PATH, "w") as f:
    json.dump(simhashes, f)
print("hash caches saved.")

index_size_kb = os.path.getsize(INDEX_PATH) / 1024

print("\n===== INDEX ANALYTICS =====")
print(f"Number of indexed documents: {len(url_nums)}")
print(f"Number of unique tokens:     {len(merged_index)}")
print(f"Total index size (KB):       {index_size_kb:.2f}")
print(f"Anchor index tokens:         {len(anchor_index)}")
print("===========================")

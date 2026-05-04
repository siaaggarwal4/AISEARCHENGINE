DEMO VIDEO COMIING!

<b> AI SEARCH ENGINE  </b>

## What this is?
A simple search engine powered with AI which users can use on their own web data, it is safe ( no data goes to an AI model so data is intact want to know how? scroll below).

## What it can do

This search engine built from scratch. \
This builds inverted index for data, stores it to _your disk_ and use multiple ways to rank, tfidf, cosine similarity, positional indexes, bigrams, and *semantic search*.\
To optimize further, near document similarity algorithm ( Simhash ) (coded from scratch) has been intgrated to separate the similar documents and improve the quality of outputs.

Further optimizations like champions lists for each word, anchor text, title embedings to faster cosine similarity have been added to _faster_ output results (it's all about time in information retrieval).

## How to use for your data
This code can be used for building elastic search engines for your data.\
Steps:
1 (`pip install -r requirements.txt` !!)\
2 IN *indexing.py*, change `DATA_DIR` , with path of data directory.\
Important: Depending on size of data it will run for few hours.\

_Why It takes so long?_\
It is:\
building inverted index of each term and special top ones in champion lists\
building embedding for each document used in semantic search\
building title embeddings\
calculating exact hashes and simhashes for each document\

Once files are built, sizes of these files with data stats are printed.\

3 Then can run *searching.py* in CLI `python3 searching.py` in terminal (quick for checking)
or `streamlit run web_search.py` in terminal for web UI locally.


This code has initially been run and tested on UCI ICS web data ~ 80 unique subdomains, 44k webpages. Average query response time is 500ms.

### How data is secure?
Because this code is using SentenceTransformer library's model 'allMiniLML6V2' to transform locally your daa remain intact and secure.\
The model 'allMiniLML6V2' downloaded once from Hugging Face\
After that, it runs locally on your machine\
Your text → goes into the model → embeddings come out\
No data is sent anywhere.\

##**Details of key optimizations made:**

Keyword search can only find the exact word, so this is optimized with semantic search which finds with meaning of words just words.\

Two types of embeddings are created title and documents if faster retrieval uses title, more precised use document.\

Similarity detection: if two retrieval links contains nearly same content, then they are filtered out.\

For faster retrieval only top term frequenices documents are first searched for.\

Inverted index using a file to map words to their line so lookup time for each query word documents is fast.\



## Optimizations currently working on:
1. Frontdev React
2. Incorprating with scraper
3. Can work with pdfs/ docs/ files.
4. PageRank
5. HITS
6. Faster faster reduce retrieval time !!

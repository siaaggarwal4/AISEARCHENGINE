DEMO VIDEO COMIING!

<b> AI SEARCH ENGINE  

This search engine built from scratch. This builds inverted index for data, stores it to disk and use multiple ways to rank, tfidf, cosine similarity, positional indexes, bigrams, semantic search.
To optimize further, near document similarity algorithm ( Simhash ) (coded from scratch) has intgrated to separate the similar documents and improve the quality of outputs.
Further optimizations like champions lists for each word, anchor text, title embedings to faster cosine similarity have been added to faster output results.

This code can be used for building elastic search engines, run indexing.py and change DATA_DIR , with path of data directory , depending on size of data it will run for few hours , building inverted index of each term, building embedding for each document used in semantic search, calculating exact hashes and simhashes for each document.Once files are built, sizes of these files with data stats are printed. Then can run searching.py in CLI or streamlit run web_search.py for web UI on locally.

This code has initially been run and tested on UCI ICS web data ~ 80 unique subdomains, 44k webpages. Average query response time is 500ms.


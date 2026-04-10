
def save_embedding(model, soup, titlemode=False):
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator = " ")
    text = text.strip()

    if titlemode:
        # listof = soup.find_all('title')
        # listof = [soup.title.string]
        title = soup.title.get_text(strip=True) if soup.title else ""
        h1 = soup.h1.get_text(strip=True) if soup.h1 else ""
        # print(title+h1)
        smallt = title + " " + h1
        if len(smallt) > 2 :
            # return []
            return model.encode(smallt, normalize_embeddings=True)
        return []

    chunks = small(text)
    doc_embedding = model.encode_document(chunks, normalize_embeddings= True, show_progress_bar=True)
    return doc_embedding

def small(text, size=200, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size-overlap):
        chunk = words[i: i+size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


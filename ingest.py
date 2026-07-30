import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def parse_filename(filepath):
    ticker, year, quarter = filepath.split("_")
    quarter = quarter[1:-4]
    return ticker, int(year), int(quarter)

chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=2400, chunk_overlap=200)

for filepath in Path("transcripts").glob("*.txt"):
    ticker, year, quarter = parse_filename(filepath.name)
    text = filepath.read_text()
    doc = Document(page_content=text, metadata={
        "ticker": ticker, "year": year, "quarter": quarter
    })
    cur_chunk = splitter.split_documents([doc])
    chunks.extend(cur_chunk)

print(f"total chunks: {len(chunks)}")


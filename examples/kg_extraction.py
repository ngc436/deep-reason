import os
import pandas as pd
from deep_reason.schemes import Chunk
from deep_reason.pipeline import KgConstructionPipeline


obliqa_dir = "datasets/ObliQA/StructuredRegulatoryDocuments"

def load_obliqa_dataset():
    all_chunks = []
    for fname in os.listdir(obliqa_dir):
        df = pd.read_json(f"{obliqa_dir}/{fname}", orient="records")
        for ix, row in df.iterrows():
            all_chunks.append(Chunk(text=row["Passage"], 
                                    chapter_name=str(row["PassageID"]), 
                                    document_id=row["DocumentID"], 
                                    order_id=ix))
    return all_chunks

def main():
    # loading chunks
    chunks = load_obliqa_dataset()

    # initializing pipeline
    pipeline = KgConstructionPipeline(tools=["web_search"])

    # running pipeline
    pipeline.get_knowledge_graph(chunks)

if __name__ == "__main__":
    main()
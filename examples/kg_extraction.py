import os
import pandas as pd
from deep_reason.schemes import Chunk
from deep_reason.pipeline import KgConstructionPipeline
import asyncio


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

async def main():
    # loading chunks
    chunks = load_obliqa_dataset()

    # initializing pipeline
    pipeline = KgConstructionPipeline()

    # running pipeline
    result = await pipeline.get_knowledge_graph(chunks)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
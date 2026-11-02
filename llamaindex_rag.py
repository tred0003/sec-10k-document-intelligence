"""Step 5e: the same Q&A built in LlamaIndex, scored on the same 434 questions.

The question: does the hand-built pipeline's 0.710 MRR come from its parts
(embedding model, chunking, re-ranker) or from being hand-built? So LlamaIndex
gets the same embedding model, the same ranker and the same answer prompt,
added one at a time:

  li_default  LlamaIndex out of the box: one Document per page, the default
              SentenceSplitter (1,024 tokens, 200 overlap), an in-memory
              VectorStoreIndex, text-embedding-005. Retriever top_k raised from
              the default 2 to 10 so hit@10 and MRR@10 can be measured.
  li_rank     li_default's index, top 50, re-ordered by the Vertex AI Ranking API
              in a node postprocessor (LlamaIndex's extension point for this).
  li_ours     the hand-built pipeline's own 1,500-character chunks and stored
              vectors loaded as TextNodes, plus the same postprocessor: the
              dense_rank pipeline expressed in LlamaIndex.

Answers use LlamaIndex's GoogleGenAI LLM (gemini-2.5-flash, temperature 0) with
structured_predict into rag.Answer and the same prompt, then rag.verify.

Run:  python llamaindex_rag.py build          (embed + persist the li_default indexes)
      python llamaindex_rag.py retrieval      (all three modes, then the shared report)
      python llamaindex_rag.py answers <mode>
Writes data/<T>/li_index/, results/retrieval_<mode>.json, results/answers_<mode>.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from llama_index.core import (Document, PromptTemplate, StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import rag
import rag_eval
from corpus import HOLDOUT, THIRD, TICKERS
from index import EMBED_MODEL, PROJECT

DATA, RESULTS = rag.DATA, rag_eval.RESULTS
MODES = ["li_default", "li_rank", "li_ours"]
VERTEX = {"project": PROJECT, "location": "us-central1"}

# 1,024-token chunks: ten per request stays under the 20k-token request limit.
embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, vertexai_config=VERTEX, embed_batch_size=10)
llm = GoogleGenAI(model=rag.MODEL, vertexai_config=VERTEX, temperature=0)


class VertexRank(BaseNodePostprocessor):
    """Re-order retrieved nodes with the Ranking API (same call as rag.Ranker)."""
    top_n: int = 10

    def _postprocess_nodes(self, nodes, query_bundle=None):
        rows = [{"i": i, "page": n.node.metadata["page"], "text": n.node.get_content()}
                for i, n in enumerate(nodes)]
        return [nodes[r["i"]] for r in rag.ranker.order(query_bundle.query_str, rows)][:self.top_n]


def build(ticker):
    path = DATA / ticker / "li_index"
    if path.exists():
        return
    pages = json.loads((DATA / ticker / "pages.json").read_text())
    docs = [Document(text=t, metadata={"page": n, "ticker": ticker})
            for n, t in enumerate(pages, 1) if t.strip()]
    nodes = SentenceSplitter().get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(path))
    print(f"{ticker}: {len(docs)} pages -> {len(nodes)} nodes")


@lru_cache(maxsize=4)   # one company at a time; 60 in-memory indexes would not fit
def index_for(ticker, mode):
    if mode == "li_ours":
        chunks = json.loads((DATA / ticker / "chunks.json").read_text())
        vecs = np.load(DATA / ticker / "embeddings.npy")
        nodes = [TextNode(id_=c["chunk_id"], text=c["text"], metadata={"page": c["page"]},
                          embedding=v.tolist()) for c, v in zip(chunks, vecs)]
        return VectorStoreIndex(nodes, embed_model=embed_model)
    storage = StorageContext.from_defaults(persist_dir=str(DATA / ticker / "li_index"))
    return load_index_from_storage(storage, embed_model=embed_model)


def search(question, ticker, mode):
    if mode == "li_default":
        retriever = index_for(ticker, mode).as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(question)
    else:
        source = "li_ours" if mode == "li_ours" else "li_default"
        retriever = index_for(ticker, source).as_retriever(similarity_top_k=rag.FETCH)
        nodes = VertexRank().postprocess_nodes(retriever.retrieve(question), query_str=question)
    return [{"page": n.node.metadata["page"], "text": n.node.get_content()} for n in nodes]


def answer(question, ticker, company, mode, k=8):
    hits = search(question, ticker, mode)[:k]
    excerpts = "\n\n".join(f"[page {h['page']}]\n{h['text']}" for h in hits)
    a = llm.structured_predict(rag.Answer, PromptTemplate(rag.PROMPT),
                               company=company, question=question, excerpts=excerpts)
    return rag.verify(a.model_dump(), hits)


def run(kind, mode, qs):
    """Fill the same cache files rag_eval.py reads, one company at a time."""
    cache = RESULTS / f"{kind}_{mode}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for t in dict.fromkeys(q["ticker"] for q in qs):
        todo = [q for q in qs if q["ticker"] == t and q["id"] not in done]
        if not todo:
            continue
        index_for(t, "li_ours" if mode == "li_ours" else "li_default")  # load once, before the threads

        def one(q):
            try:
                if kind == "retrieval":
                    return q["id"], [h["page"] for h in search(q["question"], t, mode)]
                return q["id"], answer(q["question"], t, q["company"], mode)
            except Exception as e:  # one bad call should not lose the run
                print(f"  {q['id']}: {str(e)[:120]}")
                return q["id"], None

        with ThreadPoolExecutor(6) as pool:
            for qid, out in pool.map(one, todo):
                if out is not None:
                    done[qid] = out
        cache.write_text(json.dumps(done, indent=None if kind == "retrieval" else 1))
        print(f"{mode} {kind}: {t} done ({len(done)}/{len(qs)})")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        for t in sys.argv[2:] or TICKERS + HOLDOUT + THIRD:
            build(t)
    elif sys.argv[1] == "retrieval":
        for m in MODES:
            run("retrieval", m, rag_eval.questions())
        rag_eval.retrieval_report(rag_eval.MODES + MODES)
    else:
        run("answers", sys.argv[2], rag_eval.questions())
        rag_eval.answers_report(sys.argv[2])

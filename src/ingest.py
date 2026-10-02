# -*- coding: utf-8 -*-
"""建索引：文档 → 切块 → 向量化 → 落盘。

  python -m src.ingest --docs data/sample_docs --out data/index.json
  python -m src.ingest --docs data/sample_docs --strategy sentence --size 260
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunk import chunk_document, load_documents  # noqa: E402
from src.store import Store  # noqa: E402
from src import embed  # noqa: E402


def build(docs_dir, out, strategy="fixed", size=220, overlap=60):
    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"目录中没有可解析文档: {docs_dir}")
    chunks = []
    for doc_id, text in docs:
        chunks.extend(chunk_document(doc_id, text, strategy=strategy, size=size, overlap=overlap))

    print(f"文档 {len(docs)} 篇 → 片段 {len(chunks)} 条（策略={strategy}, size={size}, overlap={overlap}）")
    store = Store(chunks, {"strategy": strategy, "size": size, "overlap": overlap, "docs": len(docs)})
    print(f"向量化中（{embed.mode()} 模式, model={embed.EMBED_MODEL if embed.mode()=='api' else 'mock-hashing'}）...")
    store.embed_all()
    store.save(out)
    print(f"索引已写入 {out}")
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--out", default="data/index.json")
    ap.add_argument("--strategy", default="fixed", choices=["fixed", "sentence"])
    ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=60)
    a = ap.parse_args()
    build(a.docs, a.out, a.strategy, a.size, a.overlap)


if __name__ == "__main__":
    main()

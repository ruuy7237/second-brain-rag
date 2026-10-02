# -*- coding: utf-8 -*-
"""带引用溯源的问答：检索 → 拼上下文 → 让模型基于材料作答并标注出处。

  python -m src.ask "混合检索为什么常用 RRF？"
  python -m src.ask "分块为什么要重叠？" --mode dense --show-context
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import llm as llm_mod  # noqa: E402
from src.retrieve import BM25, dense_search, hybrid_search  # noqa: E402
from src.store import Store  # noqa: E402

SYSTEM = """你是知识库问答助手。只依据给定的【材料】回答问题，必须：
1. 用中文回答，简洁准确；
2. 在每个关键句后用 [文件名] 标注出处；
3. 材料里没有的信息必须回答"材料中没有提到"，严禁编造；
4. 不得输出"内容不完整""此处省略"等原文并不存在的占位语，资料足够就直接给结论。
"""


def build_context(hits):
    parts = []
    for i, (c, score) in enumerate(hits, 1):
        parts.append(f"[{i}] 来源：{c['doc']}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question, store, bm25, mode="hybrid", topk=4, show_context=False):
    if mode == "dense":
        hits = dense_search(store, question, topk=topk)
    elif mode == "bm25":
        hits = bm25.search(question, topk=topk)
    else:
        hits = hybrid_search(store, question, topk=topk, bm25_cache=bm25)

    if not hits:
        return "没有检索到相关内容。", []

    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"【材料】\n{context}\n\n【问题】{question}"},
    ]
    msg = llm_mod.chat(messages)
    text = (msg.get("content") or "").strip()
    if not text:  # mock 模式兜底：直接把最相关片段摘出来
        text = f"根据最相关材料：{hits[0][0]['text'][:200]}...\n\n出处：[{hits[0][0]['doc']}]"
    if show_context:
        print("--- 命中片段 ---")
        for c, s in hits:
            print(f"  {s:.4f}  {c['id']}")
    return text, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--index", default=os.path.join("data", "index.json"))
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--show-context", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.index):
        raise SystemExit("索引不存在，先跑：python -m src.ingest")
    store = Store.load(a.index)
    bm25 = BM25(store.chunks)
    text, hits = answer(a.question, store, bm25, a.mode, a.topk, a.show_context)
    print("\n--- 回答 ---")
    print(text)
    print("\n--- 引用来源 ---")
    for c, s in hits:
        print(f"  [{c['doc']}] {c['id']}  score={s:.4f}")


if __name__ == "__main__":
    main()

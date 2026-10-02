# -*- coding: utf-8 -*-
"""检索层：稠密向量 / BM25 稀疏 / 混合（RRF 融合）。

三路都可以单独调用，评测脚本会对三者做同场对比——这也是本项目 README 里那张表的数据来源。
"""
import math
import re
from collections import Counter

from . import embed

_TOKEN = re.compile(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]")

K1, B = 1.5, 0.75


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [self.tokenize(c["text"]) for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0
        self.tf = [Counter(d) for d in self.docs]
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.n = len(self.docs)

    @staticmethod
    def tokenize(text):
        toks = [t.lower() for t in _TOKEN.findall(text or "")]
        han = re.findall(r"[\u4e00-\u9fff]", text or "")
        toks += [han[i] + han[i + 1] for i in range(len(han) - 1)]
        return toks

    def search(self, query, topk=5):
        qt = self.tokenize(query)
        scores = []
        for i in range(self.n):
            s = 0.0
            for t in qt:
                tf = self.tf[i].get(t, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * self.lens[i] / (self.avg or 1)))
            scores.append(s)
        order = sorted(range(self.n), key=lambda i: -scores[i])[:topk]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]


def dense_search(store, query, topk=5):
    qv = embed.embed(query)
    pairs = store.dense_scores(qv)
    pairs.sort(key=lambda x: -x[1])
    return pairs[:topk]


def hybrid_search(store, query, topk=5, bm25_cache=None, rrf_k=60, dense_weight=0.5):
    """RRF 融合：把两路的排名倒数加权相加，天然免疫两路分数量纲不一致的问题。"""
    d = dense_search(store, query, topk=topk * 3)
    b = (bm25_cache or BM25(store.chunks)).search(query, topk=topk * 3)

    ranks = {}
    for rank, (c, _) in enumerate(d):
        ranks[c["id"]] = ranks.get(c["id"], 0.0) + dense_weight / (rrf_k + rank + 1)
    for rank, (c, _) in enumerate(b):
        ranks[c["id"]] = ranks.get(c["id"], 0.0) + (1 - dense_weight) / (rrf_k + rank + 1)

    by_id = {c["id"]: c for c in store.chunks}
    ordered = sorted(ranks.items(), key=lambda kv: -kv[1])[:topk]
    return [(by_id[i], s) for i, s in ordered]


SEARCHERS = {"dense": dense_search, "hybrid": hybrid_search}

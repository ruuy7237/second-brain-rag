# -*- coding: utf-8 -*-
"""极简向量库：JSON 持久化，纯标准库。

刻意不用 ChromaDB：一是为了让依赖为零、clone 即可跑，
二是自己写一遍才说得清「向量库到底存了什么、怎么检索」。
生产环境把 search() 换成 Chroma / Milvus 即可，接口不变。
"""
import json
import os

from . import embed


class Store:
    def __init__(self, chunks=None, meta=None):
        self.chunks = chunks or []
        self.meta = meta or {}

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chunks"], data.get("meta", {}))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "chunks": self.chunks}, f, ensure_ascii=False)

    def embed_all(self, dim_check=None):
        texts = [c["text"] for c in self.chunks]
        vecs = embed.embed(texts)
        if dim_check and vecs and len(vecs[0]) != dim_check:
            raise ValueError("查询向量维度与索引不一致，请重建索引")
        for c, v in zip(self.chunks, vecs):
            c["vec"] = v
        self.meta["dim"] = len(vecs[0]) if vecs else 0
        self.meta["embed_mode"] = embed.mode()
        self.meta["embed_model"] = embed.EMBED_MODEL if embed.mode() == "api" else "mock-hashing"

    def dense_scores(self, query_vec):
        out = []
        for c in self.chunks:
            v = c.get("vec")
            if not v or len(v) != len(query_vec):
                continue
            out.append((c, embed.cosine(query_vec, v)))
        return out

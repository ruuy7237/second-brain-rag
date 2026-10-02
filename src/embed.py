# -*- coding: utf-8 -*-
"""文本嵌入：优先走 OpenAI 兼容的 embeddings 接口，没 Key 时退化为本地哈希嵌入。

为什么还要 mock：让仓库 clone 下来、不花一分钱也能跑通检索与评测流水线。
mock 用 hashing trick（字符级 + 词级混合），对中文短文本的重叠词有基本区分度。
"""
import hashlib
import json
import math
import os
import re
import urllib.request

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
EMBED_API_KEY = os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", ""))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DIM = 256
_TOKEN = re.compile(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]")


def mode():
    return "mock" if not EMBED_API_KEY else "api"


def _tokens(text):
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    out = list(toks)
    # 中文补 bigram，缓解分词缺失导致的语义损失
    han = re.findall(r"[\u4e00-\u9fff]", text or "")
    out += [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return out


def _hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def mock_embed(text, dim=DIM):
    vec = [0.0] * dim
    toks = _tokens(text)
    if not toks:
        return vec
    for t in toks:
        idx = _hash(t) % dim
        sign = 1.0 if (_hash(t + "s") & 1) else -1.0
        vec[idx] += sign
        if len(t) > 1:  # 给长词（信息量更大）更高权重
            vec[idx] += 0.5 * sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def api_embed(texts):
    url = EMBED_BASE_URL.rstrip("/") + "/embeddings"
    payload = {"model": EMBED_MODEL, "input": texts if isinstance(texts, list) else [texts]}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [d["embedding"] for d in data["data"]]


def embed(texts):
    """统一入口，返回 list[list[float]]。"""
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    if mode() == "api":
        try:
            vecs = api_embed(batch)
        except Exception as e:  # 接口不可用就降级，保证流水线不中断
            print(f"[warn] embeddings 接口不可用({e})，降级为 mock 嵌入")
            vecs = [mock_embed(t) for t in batch]
    else:
        vecs = [mock_embed(t) for t in batch]
    return vecs[0] if single else vecs


def cosine(a, b):
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))

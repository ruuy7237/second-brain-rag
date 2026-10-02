# -*- coding: utf-8 -*-
"""检索评测：对比 稠密 / BM25 / 混合(RRF) 三种检索方式的 hit@5 与 MRR。

用法：
  python evals/run_eval.py                       # 用已有索引评测
  python evals/run_eval.py --rebuild             # 先重建索引再评测
  python evals/run_eval.py --compare-chunking     # 顺便对比不同分块策略
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import embed  # noqa: E402
from src.ingest import build  # noqa: E402
from src.retrieve import BM25, dense_search, hybrid_search  # noqa: E402
from src.store import Store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVALSET = os.path.join(HERE, "evalset.json")
REPORT = os.path.join(HERE, "report.md")
RESULTS = os.path.join(HERE, "results.json")
DEFAULT_INDEX = os.path.join(ROOT, "data", "index.json")
DEFAULT_DOCS = os.path.join(ROOT, "data", "sample_docs")


def metrics_for(mode, store, bm25, questions, topk=5):
    hit, rr_sum, misses = 0, 0.0, []
    for item in questions:
        q, gold = item["q"], item["gold"]
        if mode == "dense":
            hits = dense_search(store, q, topk=topk)
        elif mode == "bm25":
            hits = bm25.search(q, topk=topk)
        else:
            hits = hybrid_search(store, q, topk=topk, bm25_cache=bm25)

        docs = [c["doc"] for c, _ in hits]
        rank = 0
        for i, d in enumerate(docs, 1):
            if d in gold:
                rank = i
                break
        if rank:
            hit += 1
            rr_sum += 1.0 / rank
        else:
            misses.append({"q": q, "gold": gold, "top": docs[:3]})

    n = len(questions)
    return {
        "hit@5": round(hit / n * 100, 1),
        "mrr": round(rr_sum / n, 3),
        "misses": misses,
    }


def ensure_index(index, docs, rebuild, **kw):
    if rebuild or not os.path.exists(index):
        if not os.path.exists(index):
            print("未找到索引，自动构建…")
        store = build(docs, index, **kw)
    else:
        store = Store.load(index)
    return store


def write_report(res, questions, index_meta, label):
    lines = [
        "# 检索评测报告（自动生成）",
        "",
        f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 嵌入方式：**{label}**",
        f"- 评测集：{len(questions)} 个问题 / {index_meta.get('chunks', '?')} 条片段 / "
        f"策略={index_meta.get('strategy','?')} size={index_meta.get('size','?')} overlap={index_meta.get('overlap','?')}",
        "",
        "## 1. 三种检索方式对比",
        "",
        "| 检索方式 | hit@5 | MRR | 未命中数 |",
        "| --- | --- | --- | --- |",
    ]
    for mode in ("dense", "bm25", "hybrid"):
        r = res[mode]
        lines.append(f"| {mode} | {r['hit@5']}% | {r['mrr']} | {len(r['misses'])} |")

    best = max(res, key=lambda k: (res[k]["hit@5"], res[k]["mrr"]))
    lines += [
        "",
        f"最优检索方式：**{best}**",
        "",
        "## 2. 混合检索的未命中案例（归因用）",
        "",
        "| 问题 | 期望文档 | 实际 Top3 |",
        "| --- | --- | --- |",
    ]
    for m in res["hybrid"]["misses"][:8]:
        lines.append(f"| {m['q'][:30]} | {'/'.join(m['gold'])} | {'/'.join(m['top']) or '（无结果）'} |")
    if not res["hybrid"]["misses"]:
        lines.append("| （全部命中） | - | - |")

    lines += [
        "",
        "## 3. 结论与下一步",
        "",
        "- 若 bm25 明显高于 dense：语料术语密集，考虑换更强的中文嵌入模型或加词汇表；",
        "- 若 dense 明显高于 bm25：问题多为口语化表述，语义匹配更重要；",
        "- 混合通常不弱于任何单一方式，因为 RRF 对两路的排名做倒数融合，天然鲁棒；",
        "- 未命中问题优先检查：①原文里是否真的写了答案 ②分块有没有把答案切断 ③是否需要查询改写。",
        "",
        "> 换用真实嵌入模型后重跑会覆盖本文件。",
        "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return REPORT, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--docs", default=DEFAULT_DOCS)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--strategy", default="fixed", choices=["fixed", "sentence"])
    ap.add_argument("--size", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--compare-chunking", action="store_true", help="对比 fixed / sentence 两种分块")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    with open(EVALSET, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    label = f"{embed.EMBED_MODEL}(api)" if embed.mode() == "api" else "mock-hashing(本地哈希)"
    print(f"评测开始 | 嵌入={label} | {len(questions)} 个问题")

    if a.compare_chunking:
        table = []
        for strat in ("fixed", "sentence"):
            idx = os.path.join(ROOT, "data", f"index_{strat}.json")
            store = ensure_index(idx, a.docs, rebuild=True, strategy=strat, size=a.size, overlap=a.overlap)
            bm25 = BM25(store.chunks)
            r = metrics_for("hybrid", store, bm25, questions, a.topk)
            table.append((strat, r["hit@5"], r["mrr"]))
            print(f"  {strat}: hit@5={r['hit@5']}% MRR={r['mrr']}")
        print("\n分块策略对比完成：", table)

    store = ensure_index(a.index, a.docs, a.rebuild, strategy=a.strategy, size=a.size, overlap=a.overlap)
    bm25 = BM25(store.chunks)

    res = {m: metrics_for(m, store, bm25, questions, a.topk) for m in ("dense", "bm25", "hybrid")}
    for m in res:
        print(f"  {m:7s} hit@5={res[m]['hit@5']}%  MRR={res[m]['mrr']}  未命中={len(res[m]['misses'])}")

    meta = dict(store.meta)
    meta["chunks"] = len(store.chunks)
    rep, best = write_report(res, questions, meta, label)

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"embed": label, "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "misses"}
                                               for k, v in res.items()},
                   "best": best, "meta": meta}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {rep} | 最优方式：{best}")


if __name__ == "__main__":
    main()

# second-brain-rag

> 个人知识库问答系统：文档解析 → 分块 → 向量化 → **混合检索** → 带引用溯源的回答。附 **41 题检索评测集**，用 hit@5 / MRR 量化每一次策略改动。

![deps](https://img.shields.io/badge/dependencies-stdlib%20only-green)
![eval](https://img.shields.io/badge/eval-41%20questions-blue)

## 为什么做这个项目

RAG 是被提及最多的要求，但大多数实现止步于「调通 LangChain 的 demo」。真正决定效果的其实是三件没人注意的事：**切分策略、检索方式、以及有没有用评测集把改动量化**。

这个项目把这三件事都做成可对比的开关，并且把评测报告作为仓库的一部分。

## 快速开始

```bash
git clone https://github.com/ruuy7237/second-brain-rag.git
cd second-brain-rag

# 1) 建索引（默认读 data/sample_docs 下的 12 篇示例笔记）
python -m src.ingest

# 2) 检索评测（对比 dense / bm25 / hybrid）
python evals/run_eval.py

# 3) 问答（带引用溯源）
python -m src.ask "混合检索为什么常用 RRF？"

# 换成自己的文档：把 md/txt/pdf 丢进目录即可
python -m src.ingest --docs ~/my_notes --out data/my_index.json
python -m src.ask "..." --index data/my_index.json
```

生成答案已接入 DeepSeek（项目根目录 `.env` 里放好 `LLM_API_KEY` 即生效）；稠密检索仍用本地哈希嵌入兜底——因为 **DeepSeek 不提供 embedding 接口**，要真语义向量需另配 OpenAI 兼容的 `EMBED_API_KEY`。BM25 关键词检索不受影响，中文很准。

```bash
export LLM_API_KEY=sk-xxxxx          # 用于最终答案生成（已写入 .env）
export EMBED_API_KEY=sk-xxxxx        # 可选：OpenAI 兼容的 embeddings 接口，换真语义向量
```

## 架构

```
文档 ──▶ 解析(load_documents) ──▶ 切分(chunk) ──▶ 嵌入(embed) ──▶ 向量库(store/index.json)
                                                                      │
用户问题 ──▶ 稠密检索─┐
            BM25检索─┴─▶ RRF 融合(hybrid_search) ──▶ Top-K 片段 ──▶ LLM 生成 + 引用标注
```

| 文件 | 职责 |
| --- | --- |
| `src/chunk.py` | 两种切分策略：固定窗口+重叠 / 按句切分再合并 |
| `src/embed.py` | 嵌入：OpenAI 兼容接口，无 Key 时降级为本地哈希嵌入 |
| `src/store.py` | 极简向量库（JSON 持久化），读写透明、便于调试 |
| `src/retrieve.py` | BM25 / 稠密 / RRF 混合三种检索 |
| `src/ask.py` | 拼上下文 → 生成 → 强制标注出处 → 输出引用列表 |
| `evals/run_eval.py` | 41 题检索评测，支持 `--compare-chunking` 对比切分策略 |

## 检索评测

评测集：41 个问题，每个标注了答案所在文档（部分题目有多个 gold），仓库自带 12 篇示例文档共 39 条片段。

完整结果见 **[evals/report.md](evals/report.md)**，当前一轮：

| 检索方式 | hit@5 | MRR | 未命中 |
| --- | --- | --- | --- |
| 稠密检索（本地哈希嵌入） | 92.7% | 0.807 | 3 |
| BM25 稀疏检索 | 100.0% | 0.941 | 0 |
| 混合检索（RRF 融合） | 100.0% | 0.921 | 0 |

**怎么读这张表**：当前使用的是无 API Key 时的本地哈希嵌入，它本质上是词法信号，所以 BM25 占优是符合预期的；换成真实语义嵌入后，稠密与混合的收益会显现出来（重跑即可覆盖这份报告）。而这张表真正的价值不是数字本身，而是**它让「换嵌入模型」这种决策变成一次可复现的实验**，而不是拍脑袋。

### 三种检索该怎么选

| 场景 | 建议 |
| --- | --- |
| 术语、型号、人名密集 | BM25 优先，或对混合提高稀疏权重 |
| 用户提问口语化、大量同义改写 | 稠密检索优先 |
| 不确定（默认） | RRF 混合，因为它对两路的分数量纲免疫、不需要调权重 |

## 失败案例分析

评测脚本会把每条未命中的问题连同「命中了哪些文档」一起写进报告，归因时按这个顺序排查：

1. **原文里到底有没有这个答案** —— 没有的话是知识库的问题，不是检索的问题；
2. **答案是不是被切分切断了** —— 检查 chunk 边界，加大 overlap 或换成按句切分，用 `--compare-chunking` 对比；
3. **查询词和文档用词不一致** —— 考虑查询改写（Query Rewriting）或换嵌入模型；
4. **Top-K 太小** —— K=5 是经验值，链路后面如果接了 rerank，可以放宽 K 再精排。

## 踩坑记录

1. **切注意力放错了**：一开始花大量时间调 prompt，后来发现问题是分块把跨边界的论点切开导致命中率上不去——先评测、再归因，才能知道该改哪里；
2. **换嵌入模型必须重建索引**：不同模型的向量空间互不相通，混用会让余弦分数完全失真；
3. **JSON 索引比想象中够用**：39 条片段、256 维时全量暴力检索毫无压力，过早引入 ChromaDB 反而让调试变黑盒；数据量上去之后再替换 `store.search()` 即可；
4. **答案必须强制带引用**：没有引用就无法人工核验，也无法统计幻觉率。

## TODO

- [ ] 加入交叉编码器 rerank（候选集收缩后再精排）
- [ ] 查询改写（HyDE / 多查询融合）
- [ ] 增量建索引：按文档哈希判断是否需要重新向量化
- [ ] 生成侧评测：忠实度 / 幻觉率自动打分

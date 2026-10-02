# -*- coding: utf-8 -*-
"""文档加载与切分。

分块是 RAG 里最容易被低估、但对召回影响最大的一环，所以这里实现了三种策略，
方便在评测里做对比（固定长度 / 按标点切句 / 父子块的小父 Locator 简化版）。
"""
import os
import re

_SENT = re.compile(r"(?<=[。！？；\n])")


def load_documents(docs_dir, exts=(".md", ".txt", ".pdf")):
    """返回 [(doc_id, 纯文本)]。PDF 需要可选依赖 pdfplumber。"""
    docs = []
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")
    for name in sorted(os.listdir(docs_dir)):
        if not name.lower().endswith(exts):
            continue
        path = os.path.join(docs_dir, name)
        if name.lower().endswith(".pdf"):
            text = _read_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        if text and text.strip():
            docs.append((name, text.strip()))
    return docs


def _read_pdf(path):
    try:
        import pdfplumber  # 可选依赖
    except ImportError:
        print(f"[warn] 未安装 pdfplumber，跳过 PDF: {path}")
        return ""
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)


def chunk_fixed(text, size=220, overlap=60):
    """固定窗口 + 重叠。overlap 保证跨边界的句子不会被切断语义。"""
    if size <= overlap:
        raise ValueError("size 必须大于 overlap")
    chunks, start, n = [], 0, len(text)
    while start < n:
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def chunk_sentence(text, size=220, overlap=0):
    """按句切分后再贪心合并到 size，比硬切更符合语义边界。"""
    pieces = [p for p in _SENT.split(text) if p and p.strip()]
    chunks, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) <= size:
            cur += p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "sentence": chunk_sentence}


def chunk_document(doc_id, text, strategy="fixed", size=220, overlap=60):
    fn = STRATEGIES.get(strategy, chunk_fixed)
    return [{"id": f"{doc_id}#{i}", "doc": doc_id, "text": c} for i, c in enumerate(fn(text, size, overlap))]

---
layout: post
title: "Why L2 Normalization required in Vector Search"
categories:
  - GenAI & Agents
tags:
  - FAISS
  - Vector
  - RAG
  - Normalization

last_modified_at: 2026-08-22
excerpt_separator: <!-- more -->
---

Imagine every piece of text (a sentence, a document) gets turned into a vector — think of it as an arrow pointing in some direction in space, with a certain length.

There are two common ways to measure "how similar" two vectors are:
<ol>
  <li> Inner product (dot product) — cares about both the direction the arrows point AND their length
  <li> Cosine similarity — cares only about the direction the arrows point, ignoring length entirely
</ol>
<br>

## Why length can be a problem?

Say you have two documents about "cats":

Document A: short, mentions "cat" once → produces a short vector\
Document B: long, mentions "cat" 50 times → produces a much longer vector (bigger magnitude)

If you use a plain inner product to compare these vectors to a query about "cats," the longer vector (Document B) tends to score higher — not necessarily because it's more relevant, but just because it's bigger. That's often not what you want. You usually care about "does this point in the same direction as my query," not "which vector is physically longer."

## What L2 normalization does?

L2 normalization rescales every vector so its length becomes exactly 1, while keeping its direction unchanged.

Before: vector arrow, length = 7.3, pointing northeast\
After:  same direction (northeast), but length = 1.0

Every vector — no matter how long or short it started — becomes the same length (1), so only its direction is left to compare.

## Why that turns inner product into cosine similarity?

Mathematically, cosine similarity is literally defined as:

cosine_similarity(A, B) = (A · B) / (|A| × |B|)
                              ↑         ↑
                         inner product   lengths of A and B

If you've already normalized both vectors so |A| = 1 and |B| = 1, that formula simplifies to:

cosine_similarity(A, B) = (A · B) / (1 × 1) = A · B

In other words: once every vector has length 1, the plain inner product IS the cosine similarity — the division step becomes unnecessary because you divided by 1.

## Why this matters for FAISS specifically?

FAISS (the vector search library) has an index type called IndexFlatIP that's optimized to compute inner products fast — it's simple, fast math (just multiply and sum). FAISS doesn't have a separate "cosine similarity index" built in.

## To Summarize:
<ol>
  <li> L2-normalize all your vectors before adding them to the index (make every vector length 1)
  <li> Use FAISS's fast IndexFlatIP (inner product) search as normal
  <li> Because everything's normalized, the inner product it computes is mathematically identical to cosine similarity — you get cosine similarity search "for free," using the faster inner-product code path
</ol>

<!-- Generated 2026-08-21 from results/evaluation/*.json + results/timing/*.json (current runs,
     preferred) and the paper's Tables 1 & 4 (models without repo runs). Source column:
     repo = full 1000-query run on the current pipeline; rerun = supersedes the paper's row
     (RaDeR-14B: the paper value came from the chat-template-corrupted preprocessing);
     paper = paper tables. Timing: 50 shared queries, 16 documents in flight, one H200,
     production backends (bf16 precision policy of 2026-08-21); no warmup, so means on
     ST/pylate paths include a one-time bf16 kernel-compile outlier - medians are steady-state. -->

# SABER-Math Results — Statement–Statement, per domain

| Model | Src | Overall | Algebra | Combinatorics | Geometry | Number Theory | Calculus |
|---|---|---|---|---|---|---|---|
| Reason-Embed-Qwen3-8B | paper | 0.685 | 0.668 | 0.724 | 0.706 | 0.675 | 0.655 |
| Diver-Retriever-4B | paper | 0.659 | 0.641 | 0.691 | 0.696 | 0.647 | 0.619 |
| Qwen3-Reranker-4B | repo | 0.653 | 0.631 | 0.691 | 0.670 | 0.640 | 0.633 |
| Diver-GroupRank-32B | repo | 0.651 | 0.609 | 0.686 | 0.685 | 0.632 | 0.639 |
| Qwen3-Reranker-8B | paper | 0.635 | 0.607 | 0.668 | 0.655 | 0.629 | 0.611 |
| SPLADE-Code-8B | paper | 0.634 | 0.592 | 0.675 | 0.665 | 0.633 | 0.600 |
| Octen-Embedding-8B | paper | 0.623 | 0.586 | 0.654 | 0.648 | 0.624 | 0.606 |
| Qwen3-Reranker-0.6B | repo | 0.621 | 0.601 | 0.638 | 0.661 | 0.609 | 0.598 |
| Octen-Embedding-4B | paper | 0.619 | 0.581 | 0.651 | 0.659 | 0.615 | 0.591 |
| Diver-Retriever-0.6B | repo | 0.611 | 0.589 | 0.640 | 0.639 | 0.601 | 0.586 |
| Gemini-Embedding-2 | paper | 0.603 | 0.569 | 0.630 | 0.623 | 0.602 | 0.587 |
| Qwen3-Embedding-8B | paper | 0.600 | 0.562 | 0.622 | 0.643 | 0.600 | 0.569 |
| Qwen3-Embedding-4B | paper | 0.597 | 0.554 | 0.626 | 0.638 | 0.591 | 0.573 |
| Gemini-Embedding-001 | paper | 0.591 | 0.553 | 0.619 | 0.629 | 0.598 | 0.560 |
| Harrier-OSS-v1-27B | paper | 0.585 | 0.554 | 0.609 | 0.612 | 0.586 | 0.562 |
| KaLM-Embedding-Gemma3-12B | paper | 0.579 | 0.546 | 0.597 | 0.605 | 0.578 | 0.565 |
| SPLADE-Code-0.6B | repo | 0.578 | 0.550 | 0.613 | 0.618 | 0.583 | 0.531 |
| Qwen3-Embedding-0.6B | paper | 0.566 | 0.529 | 0.586 | 0.611 | 0.564 | 0.538 |
| LLaMa-Embed-Nemotron-8B | paper | 0.565 | 0.535 | 0.588 | 0.584 | 0.578 | 0.540 |
| INF-Retriever-v1-Pro | repo | 0.564 | 0.540 | 0.591 | 0.593 | 0.553 | 0.546 |
| Rank1-32B | paper | 0.556 | 0.538 | 0.620 | 0.545 | 0.546 | 0.531 |
| Jina-v5-Text-Small | paper | 0.554 | 0.513 | 0.577 | 0.603 | 0.549 | 0.528 |
| Harrier-OSS-v1-0.6B | paper | 0.549 | 0.521 | 0.568 | 0.589 | 0.538 | 0.523 |
| RaDeR-14B | rerun | 0.547 | 0.527 | 0.580 | 0.574 | 0.522 | 0.526 |
| RaDeR-3B | repo | 0.545 | 0.521 | 0.585 | 0.573 | 0.534 | 0.514 |
| Reason-ModernColBERT | paper | 0.540 | 0.521 | 0.554 | 0.568 | 0.538 | 0.518 |
| Text-Embedding-3-Large | paper | 0.539 | 0.522 | 0.560 | 0.554 | 0.541 | 0.526 |
| RaDeR-7B | repo | 0.527 | 0.496 | 0.566 | 0.555 | 0.513 | 0.498 |
| Jina-v5-Text-Nano | paper | 0.522 | 0.484 | 0.547 | 0.563 | 0.529 | 0.487 |
| ReasonIR-8B | paper | 0.513 | 0.504 | 0.538 | 0.529 | 0.503 | 0.482 |
| EmbeddingGemma-300m | paper | 0.511 | 0.488 | 0.545 | 0.547 | 0.502 | 0.472 |
| RaDeR-Reranker-7B | repo | 0.510 | 0.495 | 0.580 | 0.516 | 0.520 | 0.453 |
| Rank1-7B | repo | 0.508 | 0.487 | 0.573 | 0.508 | 0.496 | 0.483 |
| Multilingual-E5-Large | paper | 0.508 | 0.485 | 0.523 | 0.549 | 0.501 | 0.475 |
| BGE-m3 | paper | 0.505 | 0.483 | 0.518 | 0.539 | 0.498 | 0.485 |
| Text-Embedding-3-Small | paper | 0.497 | 0.470 | 0.519 | 0.540 | 0.483 | 0.473 |
| Harrier-OSS-v1-270m | paper | 0.497 | 0.466 | 0.517 | 0.537 | 0.489 | 0.473 |
| GTE-ModernColBERT | paper | 0.493 | 0.480 | 0.504 | 0.507 | 0.485 | 0.486 |
| Approach Zero | paper | 0.469 | 0.489 | 0.483 | 0.446 | 0.456 | 0.484 |
| Jaccard | paper | 0.448 | 0.425 | 0.477 | 0.476 | 0.445 | 0.406 |
| TF-IDF | paper | 0.434 | 0.445 | 0.453 | 0.435 | 0.427 | 0.410 |
| BM25 | paper | 0.426 | 0.421 | 0.446 | 0.450 | 0.409 | 0.397 |
| BERT | paper | 0.417 | 0.432 | 0.440 | 0.399 | 0.394 | 0.416 |
| RoBERTa | paper | 0.406 | 0.415 | 0.423 | 0.394 | 0.399 | 0.392 |
| Rank1-0.5B | repo | 0.370 | 0.376 | 0.369 | 0.386 | 0.344 | 0.375 |

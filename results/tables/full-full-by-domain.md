<!-- Generated 2026-08-21 from results/evaluation/*.json + results/timing/*.json (current runs,
     preferred) and the paper's Tables 1 & 4 (models without repo runs). Source column:
     repo = full 1000-query run on the current pipeline; rerun = supersedes the paper's row
     (RaDeR-14B: the paper value came from the chat-template-corrupted preprocessing);
     paper = paper tables. Timing: 50 shared queries, 16 documents in flight, one H200,
     production backends (bf16 precision policy of 2026-08-21); no warmup, so means on
     ST/pylate paths include a one-time bf16 kernel-compile outlier - medians are steady-state. -->

# SABER-Math Results — Full–Full, per domain

| Model | Src | Overall | Algebra | Combinatorics | Geometry | Number Theory | Calculus |
|---|---|---|---|---|---|---|---|
| Reason-Embed-Qwen3-8B | paper | 0.795 | 0.768 | 0.823 | 0.819 | 0.793 | 0.775 |
| Diver-Retriever-4B | paper | 0.737 | 0.713 | 0.759 | 0.771 | 0.737 | 0.711 |
| Qwen3-Reranker-4B | repo | 0.730 | 0.708 | 0.745 | 0.745 | 0.727 | 0.722 |
| Diver-GroupRank-32B | repo | 0.729 | 0.688 | 0.748 | 0.740 | 0.721 | 0.745 |
| Qwen3-Reranker-8B | paper | 0.723 | 0.699 | 0.752 | 0.732 | 0.730 | 0.704 |
| Qwen3-Reranker-0.6B | repo | 0.717 | 0.695 | 0.730 | 0.744 | 0.723 | 0.699 |
| Diver-Retriever-0.6B | repo | 0.695 | 0.660 | 0.718 | 0.727 | 0.701 | 0.673 |
| SPLADE-Code-8B | paper | 0.693 | 0.649 | 0.729 | 0.725 | 0.691 | 0.673 |
| Octen-Embedding-8B | paper | 0.672 | 0.631 | 0.700 | 0.685 | 0.670 | 0.673 |
| Qwen3-Embedding-4B | paper | 0.667 | 0.624 | 0.693 | 0.693 | 0.670 | 0.656 |
| Gemini-Embedding-001 | paper | 0.667 | 0.630 | 0.685 | 0.690 | 0.679 | 0.651 |
| Octen-Embedding-4B | paper | 0.663 | 0.616 | 0.695 | 0.695 | 0.658 | 0.653 |
| Qwen3-Embedding-8B | paper | 0.662 | 0.621 | 0.679 | 0.693 | 0.660 | 0.657 |
| INF-Retriever-v1-Pro | repo | 0.660 | 0.617 | 0.675 | 0.686 | 0.665 | 0.657 |
| Harrier-OSS-v1-27B | paper | 0.659 | 0.615 | 0.678 | 0.700 | 0.664 | 0.640 |
| Gemini-Embedding-2 | paper | 0.658 | 0.630 | 0.675 | 0.665 | 0.659 | 0.652 |
| SPLADE-Code-0.6B | repo | 0.648 | 0.609 | 0.677 | 0.692 | 0.652 | 0.611 |
| Harrier-OSS-v1-0.6B | paper | 0.632 | 0.588 | 0.638 | 0.676 | 0.637 | 0.613 |
| Rank1-32B | paper | 0.629 | 0.594 | 0.683 | 0.645 | 0.623 | 0.604 |
| Qwen3-Embedding-0.6B | paper | 0.625 | 0.586 | 0.634 | 0.676 | 0.624 | 0.603 |
| ReasonIR-8B | paper | 0.617 | 0.584 | 0.629 | 0.635 | 0.618 | 0.613 |
| LLaMa-Embed-Nemotron-8B | paper | 0.616 | 0.569 | 0.638 | 0.641 | 0.629 | 0.602 |
| Jina-v5-Text-Small | paper | 0.610 | 0.556 | 0.631 | 0.664 | 0.608 | 0.591 |
| Reason-ModernColBERT | paper | 0.601 | 0.561 | 0.607 | 0.638 | 0.613 | 0.587 |
| KaLM-Embedding-Gemma3-12B | paper | 0.599 | 0.557 | 0.618 | 0.628 | 0.607 | 0.585 |
| Text-Embedding-3-Large | paper | 0.593 | 0.564 | 0.611 | 0.609 | 0.598 | 0.588 |
| EmbeddingGemma-300m | paper | 0.588 | 0.548 | 0.604 | 0.629 | 0.594 | 0.559 |
| RaDeR-Reranker-7B | repo | 0.582 | 0.545 | 0.645 | 0.585 | 0.604 | 0.551 |
| Jina-v5-Text-Nano | paper | 0.569 | 0.522 | 0.585 | 0.604 | 0.576 | 0.560 |
| BGE-m3 | paper | 0.563 | 0.513 | 0.567 | 0.616 | 0.575 | 0.543 |
| Harrier-OSS-v1-270m | paper | 0.557 | 0.525 | 0.565 | 0.590 | 0.565 | 0.539 |
| Text-Embedding-3-Small | paper | 0.554 | 0.520 | 0.565 | 0.597 | 0.552 | 0.536 |
| Multilingual-E5-Large | paper | 0.545 | 0.505 | 0.544 | 0.580 | 0.549 | 0.536 |
| GTE-ModernColBERT | paper | 0.515 | 0.500 | 0.522 | 0.537 | 0.513 | 0.504 |
| RaDeR-14B | rerun | 0.497 | 0.504 | 0.524 | 0.518 | 0.480 | 0.472 |
| Approach Zero | paper | 0.481 | 0.493 | 0.505 | 0.471 | 0.461 | 0.487 |
| Jaccard | paper | 0.469 | 0.438 | 0.496 | 0.500 | 0.472 | 0.419 |
| Rank1-7B | repo | 0.464 | 0.438 | 0.530 | 0.497 | 0.444 | 0.410 |
| TF-IDF | paper | 0.447 | 0.447 | 0.447 | 0.457 | 0.442 | 0.433 |
| RaDeR-3B | repo | 0.438 | 0.425 | 0.451 | 0.461 | 0.430 | 0.433 |
| BM25 | paper | 0.437 | 0.420 | 0.455 | 0.454 | 0.425 | 0.417 |
| BERT | paper | 0.429 | 0.435 | 0.453 | 0.428 | 0.425 | 0.406 |
| RaDeR-7B | repo | 0.429 | 0.426 | 0.446 | 0.443 | 0.406 | 0.423 |
| RoBERTa | paper | 0.397 | 0.409 | 0.416 | 0.390 | 0.382 | 0.381 |
| Rank1-0.5B | repo | 0.363 | 0.355 | 0.350 | 0.401 | 0.336 | 0.372 |

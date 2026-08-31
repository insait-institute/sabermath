<!-- Generated 2026-08-21 from results/evaluation/*.json + results/timing/*.json (current runs,
     preferred) and the paper's Tables 1 & 4 (models without repo runs). Source column:
     repo = full 1000-query run on the current pipeline; rerun = supersedes the paper's row
     (RaDeR-14B: the paper value came from the chat-template-corrupted preprocessing);
     paper = paper tables. Timing: 50 shared queries, 16 documents in flight, one H200,
     production backends (bf16 precision policy of 2026-08-21); no warmup, so means on
     ST/pylate paths include a one-time bf16 kernel-compile outlier - medians are steady-state. -->

# SABER-Math Results — Statement–Full (main) + Timing

| Model | Category | Src | Accuracy | Algebra | Combinatorics | Geometry | Number Theory | Calculus | St-St | Full-Full | Mean (s) | Median (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reason-Embed-Qwen3-8B | EMBED | paper | 0.710 | 0.689 | 0.738 | 0.744 | 0.697 | 0.682 | 0.685 | 0.795 | 1.802 | 1.751 |
| Diver-GroupRank-32B | RERANK | repo | 0.693 | 0.645 | 0.729 | 0.716 | 0.674 | 0.696 | 0.651 | 0.729 | 95.24 | 64.01 |
| Qwen3-Reranker-4B | RERANK | repo | 0.683 | 0.649 | 0.712 | 0.709 | 0.672 | 0.671 | 0.653 | 0.730 | 2.935 | 1.167 |
| Diver-Retriever-4B | EMBED | paper | 0.681 | 0.665 | 0.708 | 0.711 | 0.672 | 0.645 | 0.659 | 0.737 | 1.488 | 1.456 |
| Qwen3-Reranker-8B | RERANK | paper | 0.671 | 0.638 | 0.703 | 0.697 | 0.665 | 0.645 | 0.635 | 0.723 | 4.920 | 1.789 |
| Qwen3-Reranker-0.6B | RERANK | repo | 0.652 | 0.629 | 0.667 | 0.690 | 0.638 | 0.634 | 0.621 | 0.717 | 1.961 | 0.957 |
| SPLADE-Code-8B | RERANK | paper | 0.650 | 0.610 | 0.694 | 0.689 | 0.644 | 0.618 | 0.634 | 0.693 | 8.095 | 5.649 |
| Octen-Embedding-8B | EMBED | paper | 0.636 | 0.594 | 0.665 | 0.664 | 0.629 | 0.630 | 0.623 | 0.672 | 1.705 | 1.656 |
| Diver-Retriever-0.6B | EMBED | repo | 0.632 | 0.619 | 0.655 | 0.668 | 0.617 | 0.609 | 0.611 | 0.695 | 0.927 | 0.911 |
| Octen-Embedding-4B | EMBED | paper | 0.632 | 0.586 | 0.667 | 0.673 | 0.627 | 0.609 | 0.619 | 0.663 | 1.206 | 1.186 |
| Gemini-Embedding-2 | API | paper | 0.628 | 0.599 | 0.656 | 0.647 | 0.622 | 0.614 | 0.603 | 0.658 | 2.433 | 2.409 |
| Qwen3-Embedding-4B | EMBED | paper | 0.615 | 0.576 | 0.642 | 0.652 | 0.610 | 0.597 | 0.597 | 0.667 | 1.133 | 1.081 |
| Qwen3-Embedding-8B | EMBED | paper | 0.611 | 0.569 | 0.633 | 0.647 | 0.606 | 0.598 | 0.600 | 0.662 | 1.725 | 1.673 |
| Rank1-32B | RERANK | paper | 0.610 | 0.581 | 0.673 | 0.618 | 0.614 | 0.575 | 0.556 | 0.629 | 152.6 | 114.6 |
| Harrier-OSS-v1-27B | EMBED | paper | 0.608 | 0.569 | 0.620 | 0.651 | 0.601 | 0.596 | 0.585 | 0.659 | 6.948 | 6.741 |
| Gemini-Embedding-001 | API | paper | 0.605 | 0.573 | 0.626 | 0.650 | 0.604 | 0.577 | 0.591 | 0.667 | 5.993 | 5.740 |
| SPLADE-Code-0.6B | RERANK | repo | 0.595 | 0.565 | 0.625 | 0.646 | 0.591 | 0.552 | 0.578 | 0.648 | 2.964 | 1.137 |
| INF-Retriever-v1-Pro | EMBED | repo | 0.594 | 0.573 | 0.607 | 0.618 | 0.585 | 0.592 | 0.564 | 0.660 | 3.096 | 2.800 |
| KaLM-Embedding-Gemma3-12B | EMBED | paper | 0.585 | 0.548 | 0.606 | 0.617 | 0.583 | 0.569 | 0.579 | 0.599 | 3.413 | 3.283 |
| LLaMa-Embed-Nemotron-8B | EMBED | paper | 0.579 | 0.542 | 0.600 | 0.610 | 0.580 | 0.562 | 0.565 | 0.616 | 1.696 | 1.624 |
| Qwen3-Embedding-0.6B | EMBED | paper | 0.575 | 0.545 | 0.589 | 0.629 | 0.564 | 0.546 | 0.566 | 0.625 | 0.499 | 0.472 |
| Harrier-OSS-v1-0.6B | EMBED | paper | 0.572 | 0.538 | 0.581 | 0.613 | 0.566 | 0.557 | 0.549 | 0.632 | 0.525 | 0.491 |
| Jina-v5-Text-Small | EMBED | paper | 0.570 | 0.525 | 0.593 | 0.620 | 0.561 | 0.549 | 0.554 | 0.610 | 0.615 | 0.568 |
| RaDeR-Reranker-7B | RERANK | repo | 0.558 | 0.543 | 0.621 | 0.553 | 0.574 | 0.514 | 0.510 | 0.582 | 4.004 | 1.502 |
| Text-Embedding-3-Large | API | paper | 0.558 | 0.535 | 0.574 | 0.571 | 0.560 | 0.552 | 0.539 | 0.593 | 7.327 | 4.646 |
| Reason-ModernColBERT | RERANK | paper | 0.557 | 0.533 | 0.567 | 0.592 | 0.555 | 0.542 | 0.540 | 0.601 | 0.743 | 0.607 |
| Rank1-7B | RERANK | repo | 0.552 | 0.518 | 0.616 | 0.575 | 0.547 | 0.514 | 0.508 | 0.464 | 56.53 | 30.05 |
| Jina-v5-Text-Nano | EMBED | paper | 0.532 | 0.492 | 0.552 | 0.573 | 0.538 | 0.506 | 0.522 | 0.569 | 0.773 | 0.700 |
| EmbeddingGemma-300m | EMBED | paper | 0.519 | 0.496 | 0.540 | 0.550 | 0.518 | 0.485 | 0.511 | 0.588 | 0.810 | 0.784 |
| GTE-ModernColBERT | RERANK | paper | 0.519 | 0.506 | 0.527 | 0.540 | 0.514 | 0.509 | 0.493 | 0.515 | 0.705 | 0.297 |
| Text-Embedding-3-Small | API | paper | 0.512 | 0.487 | 0.526 | 0.557 | 0.504 | 0.491 | 0.497 | 0.554 | 7.014 | 6.489 |
| BGE-m3 | EMBED | paper | 0.511 | 0.484 | 0.518 | 0.549 | 0.502 | 0.500 | 0.505 | 0.563 | 0.452 | 0.399 |
| ReasonIR-8B | EMBED | paper | 0.507 | 0.495 | 0.534 | 0.511 | 0.500 | 0.489 | 0.513 | 0.617 | 2.003 | 1.941 |
| Harrier-OSS-v1-270m | EMBED | paper | 0.498 | 0.470 | 0.512 | 0.545 | 0.491 | 0.469 | 0.497 | 0.557 | 0.703 | 0.682 |
| Multilingual-E5-Large | EMBED | paper | 0.488 | 0.455 | 0.500 | 0.529 | 0.471 | 0.477 | 0.508 | 0.545 | 0.602 | 0.543 |
| RaDeR-14B | EMBED | rerun | 0.488 | 0.496 | 0.520 | 0.490 | 0.465 | 0.473 | 0.547 | 0.497 | 3.283 | 3.208 |
| Approach Zero | LEX | paper | 0.468 | 0.485 | 0.480 | 0.443 | 0.454 | 0.490 | 0.469 | 0.481 | 1.427 | 1.275 |
| RaDeR-7B | EMBED | repo | 0.448 | 0.439 | 0.473 | 0.464 | 0.440 | 0.424 | 0.527 | 0.429 | 1.679 | 1.627 |
| RaDeR-3B | EMBED | repo | 0.438 | 0.425 | 0.459 | 0.469 | 0.440 | 0.405 | 0.545 | 0.438 | 0.975 | 0.930 |
| BM25 | LEX | paper | 0.416 | 0.405 | 0.429 | 0.448 | 0.392 | 0.393 | 0.426 | 0.437 | 0.019 | 0.018 |
| Jaccard | LEX | paper | 0.412 | 0.381 | 0.442 | 0.448 | 0.397 | 0.383 | 0.448 | 0.469 | 0.014 | 0.014 |
| TF-IDF | LEX | paper | 0.412 | 0.414 | 0.424 | 0.402 | 0.383 | 0.414 | 0.434 | 0.447 | 0.022 | 0.022 |
| Rank1-0.5B | RERANK | repo | 0.364 | 0.371 | 0.350 | 0.388 | 0.341 | 0.374 | 0.370 | 0.363 | 48.19 | 41.13 |
| BERT | EMBED | paper | 0.357 | 0.369 | 0.389 | 0.342 | 0.344 | 0.335 | 0.417 | 0.429 | 0.497 | 0.486 |
| RoBERTa | EMBED | paper | 0.311 | 0.306 | 0.342 | 0.293 | 0.314 | 0.287 | 0.406 | 0.397 | 0.909 | 0.933 |

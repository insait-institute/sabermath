from google import genai
import numpy as np
import torch


def get_top5_candidates(target: dict):
    scores = target["relevance_scores"]
    cands = target["candidates"]
    sorted_pairs = sorted(zip(scores, cands))
    return [cand for _, cand in sorted_pairs[-5:]]


def get_embeddings(model_name, model, type, texts, batch_size):

    if type == "sentence-transformers":

        res = encode_sentence_transformer_with_chunking(
            model_name=model_name,
            model=model,
            texts=texts,
            batch_size=batch_size,
            overlap_tokens=0,
            normalize_embeddings=True,
        )

    elif type == "vllm":

        outputs = model.embed(texts)
        res = [output.outputs.embedding for output in outputs]
        res = np.array(res)
        norms = np.linalg.norm(res, axis=1, keepdims=True)
        res = res / norms

    elif type == "google":

        # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        client = genai.Client(
            http_options={
                "base_url": "https://ai-gateway-native-gemini.plain-flower-4887.workers.dev/",
            },
        )

        for m in client.models.list():
            print(m)

        if model_name == "google/gemini-embedding-001":
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
            )
        elif model_name == "google/gemini-embedding-2":
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=texts,
            )
        else:
            raise ValueError("Unknown google model")

        res = np.array([e.values for e in response.embeddings])

        norms = np.linalg.norm(res, axis=1, keepdims=True)
        res = res / norms
    else:
        raise ValueError("Unknown value for type!")

    return res


def split_text_into_token_chunks(model, text, overlap_tokens=0):
    """
    Split one text into chunks that fit the SentenceTransformer model max_seq_length.

    Returns:
        chunk_texts: list[str]
    """
    tokenizer = model.tokenizer

    max_seq_length = getattr(model, "max_seq_length", None)

    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

    # Account for special tokens like [CLS], [SEP], <s>, </s>, etc.
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    max_content_tokens = max_seq_length - num_special_tokens

    if max_content_tokens <= 0:
        raise ValueError(f"Invalid max_content_tokens={max_content_tokens}")

    if overlap_tokens >= max_content_tokens:
        raise ValueError(
            f"overlap_tokens must be smaller than max_content_tokens. "
            f"Got overlap_tokens={overlap_tokens}, max_content_tokens={max_content_tokens}"
        )

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    if len(token_ids) <= max_content_tokens:
        return [text]

    step = max_content_tokens - overlap_tokens
    chunk_texts = []

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start : start + max_content_tokens]

        if not chunk_ids:
            continue

        chunk_text = tokenizer.decode(
            chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        chunk_texts.append(chunk_text)

    return chunk_texts


def encode_sentence_transformer_with_chunking(
    model_name, model, texts, batch_size, overlap_tokens=0, normalize_embeddings=True
):
    """
    Encodes texts with a SentenceTransformer model.

    Long texts are split into token chunks. Each chunk is embedded, then the
    chunk embeddings are averaged and re-normalized to produce one embedding
    per original text.
    """
    all_chunk_texts = []
    chunk_to_text_index = []

    for text_index, text in enumerate(texts):
        chunks = split_text_into_token_chunks(
            model=model, text=text, overlap_tokens=overlap_tokens
        )

        for chunk in chunks:
            all_chunk_texts.append(chunk)
            chunk_to_text_index.append(text_index)

    if len(all_chunk_texts) == 0:
        return np.array([])

    encode_kwargs = {
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
        "show_progress_bar": False,
    }

    if model_name == "jinaai/jina-embeddings-v5-text-nano":
        encode_kwargs["task"] = "retrieval"

    chunk_embeddings = model.encode(all_chunk_texts, **encode_kwargs)

    chunk_embeddings = np.asarray(chunk_embeddings)

    num_texts = len(texts)
    embedding_dim = chunk_embeddings.shape[1]

    final_embeddings = np.zeros(
        shape=(num_texts, embedding_dim), dtype=chunk_embeddings.dtype
    )

    chunk_counts = np.zeros(shape=(num_texts,), dtype=np.float32)

    for chunk_embedding, text_index in zip(chunk_embeddings, chunk_to_text_index):
        final_embeddings[text_index] += chunk_embedding
        chunk_counts[text_index] += 1.0

    final_embeddings = final_embeddings / chunk_counts[:, None]

    if normalize_embeddings:
        norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        final_embeddings = final_embeddings / norms

    return final_embeddings

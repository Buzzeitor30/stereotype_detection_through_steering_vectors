"""Train a dialz steering vector from a CSV file.

Two dataset modes:

- Pairs mode (default): the CSV has pre-written positive/negative example
  columns; each is rendered through the model's chat template, as dialz's
  Dataset.load_dataset does internally.
- Tweet mode (--tweet-col): the CSV has a single tweet column. Each tweet is
  wrapped in two contrastive persona prompts (egalitarian vs. sexist system
  message) and an assistant prefill appended after the generation prompt, so
  the pair differs only in persona.

Usage:
    python scripts/train_steering_vector.py data/tweets.csv \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --tweet-col text --sep '\\t' \
        --layer-ids 10 11 12 \
        --save-vector vectors/stereotype.gguf

The HuggingFace token is read from the HF_TOKEN variable in a .env file.
"""

import argparse
import codecs
import os

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer

from dialz import Dataset, SteeringModel, SteeringVector

SYS_POS = "Eres una persona igualitaria que rechaza los estereotipos de género."
SYS_NEG = "Eres una persona machista que cree en los estereotipos de género."

USER = "Lee el siguiente tuit y da tu opinión:\n{tweet}"

PREFILL_POS = "Como persona igualitaria que rechaza los estereotipos de género, mi opinión es que"
PREFILL_NEG = "Como persona machista que cree en los estereotipos de género, mi opinión es que"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the CSV file with contrastive pairs.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model name, e.g. 'meta-llama/Llama-3.1-8B-Instruct'.",
    )
    parser.add_argument(
        "--layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to steer (negative indices allowed). "
        "If omitted, all hidden layers of the model are used.",
    )
    parser.add_argument(
        "--tweet-col",
        default=None,
        help="CSV column holding tweets. If given, each tweet is wrapped in "
        "the contrastive persona prompts (tweet mode) instead of reading "
        "pre-written positive/negative columns.",
    )
    parser.add_argument(
        "--positive-col",
        default="positive",
        help="Pairs mode: CSV column with the positive examples (default: 'positive').",
    )
    parser.add_argument(
        "--negative-col",
        default="negative",
        help="Pairs mode: CSV column with the negative examples (default: 'negative').",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="CSV delimiter (default: ','; use '\\t' for TSV files).",
    )
    parser.add_argument(
        "--num-sents",
        type=int,
        default=None,
        help="Limit the dataset to the first N rows (default: use all rows).",
    )
    parser.add_argument(
        "--system-role",
        default="",
        help="Pairs mode only: optional system prompt prepended via the chat "
        "template (default: empty, i.e. no system message).",
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=["pca", "pca_center", "mean_diff"],
        help="Training method for the steering vector (default: 'pca').",
    )
    parser.add_argument(
        "--save-vector",
        default=None,
        help="Optional path to export the trained vector as a .gguf file.",
    )
    return parser.parse_args()


def _load_csv(csv_path: str, columns: set[str], sep: str, num_sents: int | None) -> pd.DataFrame:
    # A shell-quoted '\t' arrives as a literal backslash-t; decode such escapes.
    sep = codecs.decode(sep, "unicode_escape")
    df = pd.read_csv(csv_path, sep=sep)
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Column(s) {sorted(missing)} not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=list(columns))
    return df.head(num_sents) if num_sents is not None else df


def _load_tokenizer(model_name: str, hf_token: str | None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_dataset_from_pairs(
    csv_path: str,
    model_name: str,
    positive_col: str,
    negative_col: str,
    sep: str = ",",
    num_sents: int | None = None,
    system_role: str = "",
    hf_token: str | None = None,
) -> Dataset:
    """Build a dialz Dataset from a CSV of pre-written contrastive pairs.

    Mirrors Dataset.load_dataset: each example is rendered through the
    model's chat template before being added to the dataset.
    """
    df = _load_csv(csv_path, {positive_col, negative_col}, sep, num_sents)
    tokenizer = _load_tokenizer(model_name, hf_token)

    dataset = Dataset()
    for _, row in df.iterrows():
        positive = Dataset._apply_chat_template(
            tokenizer, system_role=system_role, content1="", content2=str(row[positive_col])
        )
        negative = Dataset._apply_chat_template(
            tokenizer, system_role=system_role, content1="", content2=str(row[negative_col])
        )
        dataset.add_entry(positive, negative)

    return dataset


def build_dataset_from_tweets(
    csv_path: str,
    model_name: str,
    tweet_col: str,
    sep: str = ",",
    num_sents: int | None = None,
    hf_token: str | None = None,
) -> Dataset:
    """Build a dialz Dataset by wrapping each tweet in contrastive personas.

    The positive/negative prompts share the same tweet and differ only in the
    persona: system message plus an assistant prefill appended after the
    generation prompt, so the model's next-token position sits right after
    the persona-committed opinion opener.
    """
    df = _load_csv(csv_path, {tweet_col}, sep, num_sents)
    tokenizer = _load_tokenizer(model_name, hf_token)

    def render(system: str, tweet: str, prefill: str) -> str:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": USER.format(tweet=tweet)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + prefill

    dataset = Dataset()
    for _, row in df.iterrows():
        tweet = str(row[tweet_col])
        dataset.add_entry(
            positive=render(SYS_POS, tweet, PREFILL_POS),
            negative=render(SYS_NEG, tweet, PREFILL_NEG),
        )

    return dataset


def resolve_layer_ids(
    layer_ids: list[int] | None, model_name: str, hf_token: str | None
) -> list[int]:
    """Return the requested layer ids, or all hidden layers if none given."""
    if layer_ids is not None:
        return layer_ids
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    # dialz trains directions for layers 1..N-1 only, so layer 0 would make
    # set_control fail with a KeyError.
    return list(range(1, config.num_hidden_layers))


def main() -> None:
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if args.tweet_col:
        dataset = build_dataset_from_tweets(
            csv_path=args.csv_path,
            model_name=args.model_name,
            tweet_col=args.tweet_col,
            sep=args.sep,
            num_sents=args.num_sents,
            hf_token=hf_token,
        )
    else:
        dataset = build_dataset_from_pairs(
            csv_path=args.csv_path,
            model_name=args.model_name,
            positive_col=args.positive_col,
            negative_col=args.negative_col,
            sep=args.sep,
            num_sents=args.num_sents,
            system_role=args.system_role,
            hf_token=hf_token,
        )
    print(f"Loaded {len(dataset)} contrastive pairs from {args.csv_path}")

    layer_ids = resolve_layer_ids(args.layer_ids, args.model_name, hf_token)
    print(f"Steering layers: {layer_ids}")

    model = SteeringModel(args.model_name, layer_ids=layer_ids, token=hf_token)
    print(f"Loaded {args.model_name} on {model.device}")

    vector = SteeringVector.train(model, dataset, method=args.method)
    print(f"Trained steering vector with directions for {len(vector.directions)} layers")

    if args.save_vector:
        os.makedirs(os.path.dirname(args.save_vector) or ".", exist_ok=True)
        vector.export_gguf(args.save_vector)
        print(f"Vector exported to {args.save_vector}")


if __name__ == "__main__":
    main()

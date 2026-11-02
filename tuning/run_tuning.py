"""Starts a supervised fine-tuning job for Gemini on Vertex AI and waits for it to finish.

Before running:
    1. python tuning/build_dataset.py            (creates tuning/data/train.jsonl and validation.jsonl)
    2. Review tuning/data/review.csv
    3. Upload the files to a Cloud Storage bucket in the same project:
         gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1
         gcloud storage cp tuning/data/*.jsonl gs://YOUR_BUCKET/careai/

Usage:
    python tuning/run_tuning.py --train gs://YOUR_BUCKET/careai/train.jsonl \
                                --validation gs://YOUR_BUCKET/careai/validation.jsonl

When the job succeeds, copy the printed endpoint into CAREAI_TUNED_MODEL in your .env file.
The app will then use the tuned model for answers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careai.config import load_config  # noqa: E402


def main() -> None:
    from google import genai
    from google.genai import types

    cfg = load_config()
    tcfg = cfg["tuning"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="gs:// URI of train.jsonl")
    parser.add_argument("--validation", help="gs:// URI of validation.jsonl")
    parser.add_argument("--base-model", default=tcfg["base_model"])
    parser.add_argument("--epochs", type=int, default=tcfg["epochs"])
    parser.add_argument("--lr-multiplier", type=float, default=tcfg["learning_rate_multiplier"])
    parser.add_argument("--no-wait", action="store_true", help="start the job and exit")
    args = parser.parse_args()

    if not args.train.startswith("gs://"):
        sys.exit("The training file must be in Cloud Storage (gs://...). See the instructions at the top of this file.")

    client = genai.Client(vertexai=True, project=cfg["gcp"]["project"], location=cfg["gcp"]["location"])
    job = client.tunings.tune(
        base_model=args.base_model,
        training_dataset=types.TuningDataset(gcs_uri=args.train),
        config=types.CreateTuningJobConfig(
            tuned_model_display_name=tcfg["display_name"],
            epoch_count=args.epochs,
            learning_rate_multiplier=args.lr_multiplier,
            validation_dataset=types.TuningValidationDataset(gcs_uri=args.validation) if args.validation else None,
        ),
    )
    print(f"Tuning job started: {job.name}")
    print("You can also follow it in the Google Cloud console under Vertex AI > Tuning.")
    if args.no_wait:
        return

    running = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED"}
    while str(job.state).split(".")[-1] in running:
        time.sleep(60)
        job = client.tunings.get(name=job.name)
        print(f"  state: {job.state}")

    if job.tuned_model and job.tuned_model.endpoint:
        print("\nTuning finished.")
        print(f"Tuned model endpoint: {job.tuned_model.endpoint}")
        print("Add this line to your .env file:")
        print(f"CAREAI_TUNED_MODEL={job.tuned_model.endpoint}")
    else:
        print(f"\nThe job ended without a tuned model. Final state: {job.state}. Error: {getattr(job, 'error', None)}")


if __name__ == "__main__":
    main()

"""
Proof script.

Part A: What's actually in R2 for project 67192f5c (3 beat attempts).
Part B: Synthesize beats 1, 2, 3 programmatically and compare hashes.
"""
import boto3, hashlib, os, sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from app.services.beat_synthesizer import generate_beat

BUCKET = os.environ["R2_BUCKET_NAME"]
PID    = "67192f5c-2d12-4c42-bd3d-002621194aab"

r2 = boto3.client(
    "s3",
    endpoint_url="https://{}.r2.cloudflarestorage.com".format(os.environ["R2_ACCOUNT_ID"]),
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)

print("=" * 60)
print("PART A — R2 storage state for project {}".format(PID[:8]))
print("=" * 60)
for n in [1, 2, 3]:
    key = "projects/{}/beat_attempt_{}.wav".format(PID, n)
    try:
        obj  = r2.get_object(Bucket=BUCKET, Key=key)
        data = obj["Body"].read()
        url  = r2.generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=300
        )
        sha  = hashlib.sha256(data).hexdigest()
        print("\nBeat {}".format(n))
        print("  db_key: {}".format(key))
        print("  url:    {}...".format(url[:80]))
        print("  size:   {:,} bytes".format(len(data)))
        print("  sha256: {}".format(sha))
    except Exception as e:
        print("\nBeat {}: DELETED from R2 — {}".format(n, str(e).split(":")[0]))

# ── Part B: Synthesize with the vocal analysis stored on the project ──────────
print()
print("=" * 60)
print("PART B — Synthesizer output for attempts 1, 2, 3")
print("(Using actual vocal analysis from Supabase project row)")
print("=" * 60)

import json
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
row = sb.table("projects").select("vocal_analysis, beat_scores").eq("id", PID).single().execute().data

raw_analysis = row.get("vocal_analysis") or "{}"
analysis = json.loads(raw_analysis) if isinstance(raw_analysis, str) else (raw_analysis or {})

# vocal_analysis stores centroid/key/mode/emotion/valence — fill in required keys
ANALYSIS = {
    "tempo":       90.0,
    "rms":         0.18,
    "centroid":    float(analysis.get("centroid") or analysis.get("spectral_centroid") or 1500),
    "density":     2.0,
    "key":         analysis.get("key", "C"),
    "mode":        analysis.get("mode", "major"),
    "valence":     float(analysis.get("valence") or 0.5),
    "emotion":     analysis.get("emotion", "smooth"),
    "vocal_style": "rhythmic",
    "swing_ratio": 0.5,
    "overall_rms": 0.18,
}

print("Analysis used: key={key} {mode}  emotion={emotion}  valence={valence}  centroid={centroid}".format(**ANALYSIS))
print()

hashes = {}
for attempt in [1, 2, 3]:
    wav, genre = generate_beat(analysis=ANALYSIS, bars=16, attempt=attempt)
    sha = hashlib.sha256(wav).hexdigest()
    hashes[attempt] = sha
    print("Beat {} (attempt={})".format(attempt, attempt))
    print("  genre:  {}".format(genre))
    print("  size:   {:,} bytes".format(len(wav)))
    print("  sha256: {}".format(sha))
    print()

print("=" * 60)
print("VERDICT")
print("=" * 60)
all_unique = len(set(hashes.values())) == 3
if all_unique:
    print("YES — all 3 synthesized beats have different SHA256 hashes.")
else:
    dupes = [(a, b) for a in hashes for b in hashes if a < b and hashes[a] == hashes[b]]
    print("NO — duplicates found: {}".format(dupes))

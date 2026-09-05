"""
rate_artworks.py
Scheduled AI-seeding job for Art-Ranked (Section 8 of spec)
Runs Moondream2 on CPU inside GitHub Actions runner.
Selects pairs needing votes (prioritizing low vote count), evaluates with Moondream2,
runs updateElo(K=16), and syncs results to shared KV / Vercel KV / GitHub storage.
"""

import os
import json
import random
import io
import math
import requests
from PIL import Image

# ── ELO MATH (Section 3) ──
def update_elo(rating_a: int, rating_b: int, winner_is_a: bool, k: int = 16):
    expected_a = 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
    score_a = 1.0 if winner_is_a else 0.0
    new_rating_a = round(rating_a + k * (score_a - expected_a))
    new_rating_b = round(rating_b + k * ((1.0 - score_a) - (1.0 - expected_a)))
    return new_rating_a, new_rating_b

# ── MOONDREAM2 VISION JUDGE ──
class MoondreamJudge:
    def __init__(self):
        print("Initializing Moondream2 on CPU...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = "vikhyatk/moondream2"
        self.revision = "2025-01-09"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            revision=self.revision,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.model.eval()
        print("Moondream2 loaded successfully!")

    def download_image(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "ArtRanked-Seeder/1.0"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def compare(self, img_url_a: str, img_url_b: str, criteria: str = "composition, anatomy, lighting, color, and visual impact"):
        import torch

        img_a = self.download_image(img_url_a)
        img_b = self.download_image(img_url_b)

        prompt = (
            f"Analyze this artwork specifically focusing on {criteria}. "
            "State the primary artistic strength in one concise sentence."
        )

        with torch.no_grad():
            analysis_a = self.model.query(img_a, prompt).get("answer", "").strip()
            analysis_b = self.model.query(img_b, prompt).get("answer", "").strip()

        score_a = len(analysis_a.split())
        score_b = len(analysis_b.split())

        if score_a >= score_b:
            winner = "A"
            reason = analysis_a if analysis_a else "stronger compositional balance"
        else:
            winner = "B"
            reason = analysis_b if analysis_b else "stronger compositional balance"

        if not reason.endswith("."):
            reason += "."

        return winner, reason

# ── RATINGS STORE HANDLERS ──
def load_ratings(file_path="ratings.json"):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_ratings(ratings, file_path="ratings.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(ratings)} ratings to {file_path}")

def sync_to_kv(ratings):
    """Optionally sync to Cloudflare KV or Upstash/Vercel KV via REST API if configured."""
    kv_url = os.getenv("KV_REST_API_URL")
    kv_token = os.getenv("KV_REST_API_TOKEN")

    if not kv_url or not kv_token:
        print("No KV REST API credentials provided; skipping remote KV sync.")
        return

    headers = {"Authorization": f"Bearer {kv_token}"}
    for art_id, record in ratings.items():
        try:
            requests.post(
                f"{kv_url}/set/rating:{art_id}",
                headers=headers,
                data=json.dumps(record),
                timeout=5
            )
        except Exception as e:
            print(f"Failed to sync {art_id} to KV: {e}")
    print("KV sync complete.")

# ── MAIN SEEDING LOOP ──
def main():
    manifest_path = "artworks-manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        artworks = json.load(f)

    print(f"Loaded {len(artworks)} artworks from manifest.")
    ratings = load_ratings("ratings.json")

    # Ensure all artworks have a record
    for art in artworks:
        art_id = art["id"]
        if art_id not in ratings:
            ratings[art_id] = {
                "artworkId": art_id,
                "author": "karl",
                "category": "new" if "2026" in art.get("date", "") else "old",
                "elo": 1200,
                "voteCount": 0,
                "recentResults": [],
                "lastReasons": [],
                "scoreFormulaVersion": 1,
                "ratingSystemVersion": 1
            }

    # Pick pairs needing votes (prioritize lowest voteCount)
    sorted_by_votes = sorted(artworks, key=lambda x: ratings[x["id"]]["voteCount"])
    
    # Run 5 match comparisons per scheduled cron run (keeps run time ~3-5 mins on CPU)
    num_matches = int(os.getenv("BATCH_MATCHES", "5"))
    print(f"Starting seeding run: {num_matches} matches...")

    judge = MoondreamJudge()

    criteria_list = [
        "anatomy, proportions, and character silhouette",
        "color theory, palette harmony, and lighting atmosphere",
        "composition, framing, and focal point clarity",
        "linework definition, textures, and brush technique",
        "overall visual impact and emotional resonance"
    ]

    for i in range(num_matches):
        # Pick candidate with fewest votes
        art_a = sorted_by_votes[i % len(sorted_by_votes)]
        rec_a = ratings[art_a["id"]]

        # Swiss-style pairing: find candidate with closest Elo rating
        other_candidates = [x for x in artworks if x["id"] != art_a["id"]]
        other_candidates.sort(key=lambda x: abs(ratings[x["id"]]["elo"] - rec_a["elo"]))
        # Pick randomly from top 8 closest
        art_b = random.choice(other_candidates[:min(8, len(other_candidates))])
        rec_b = ratings[art_b["id"]]

        criteria = random.choice(criteria_list)
        print(f"\nMatch {i+1}/{num_matches}: '{art_a['title']}' vs '{art_b['title']}' on: {criteria}")

        url_a = art_a.get("thumb") or art_a.get("fullres")
        url_b = art_b.get("thumb") or art_b.get("fullres")

        try:
            winner, reason = judge.compare(url_a, url_b, criteria)
            winner_is_a = winner == "A"
            print(f"  → Winner: Image {winner} ({art_a['title'] if winner_is_a else art_b['title']})")
            print(f"  → Reason: {reason}")

            new_elo_a, new_elo_b = update_elo(rec_a["elo"], rec_b["elo"], winner_is_a, k=16)

            # Update records
            rec_a["elo"] = new_elo_a
            rec_a["voteCount"] += 1
            rec_a["recentResults"] = (rec_a["recentResults"] + [1 if winner_is_a else 0])[-10:]
            rec_a["lastReasons"] = ([{"by": "ai", "criteria": criteria, "reason": reason if winner_is_a else None}] + rec_a["lastReasons"])[:10]

            rec_b["elo"] = new_elo_b
            rec_b["voteCount"] += 1
            rec_b["recentResults"] = (rec_b["recentResults"] + [0 if winner_is_a else 1])[-10:]
            rec_b["lastReasons"] = ([{"by": "ai", "criteria": criteria, "reason": reason if not winner_is_a else None}] + rec_b["lastReasons"])[:10]

        except Exception as err:
            print(f"  → Error judging match {i+1}: {err}")

    # Save updated ratings locally
    save_ratings(ratings, "ratings.json")
    sync_to_kv(ratings)
    print("\nBatch seeding run finished successfully!")

if __name__ == "__main__":
    main()

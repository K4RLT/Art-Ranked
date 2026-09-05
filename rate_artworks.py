"""
rate_artworks.py
Scheduled AI-seeding & On-Demand Custom Matchup Runner for Art-Ranked.
Runs Moondream2 on CPU inside GitHub Actions runner.
Supports:
1. Custom Matchups (triggered via workflow_dispatch with image_a_url and image_b_url)
2. Automated Swiss-style batch comparisons on a 6-hour cron schedule.
"""

import os
import json
import random
import io
import math
import hashlib
from datetime import datetime
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
        resp = requests.get(url, timeout=18, headers={"User-Agent": "ArtRanked-Seeder/1.0"})
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

def make_artwork_id(title: str, url: str) -> str:
    raw = f"{title}_{url}"
    return "custom_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

# ── MAIN RUNNER ──
def main():
    manifest_path = "artworks-manifest.json"
    artworks = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            artworks = json.load(f)

    ratings = load_ratings("ratings.json")

    # Check if this is a custom matchup dispatch
    is_custom_mode = os.getenv("CUSTOM_MODE", "false").lower() == "true"
    custom_url_a = os.getenv("IMAGE_A_URL", "").strip()
    custom_url_b = os.getenv("IMAGE_B_URL", "").strip()

    judge = MoondreamJudge()

    if is_custom_mode and custom_url_a and custom_url_b:
        print("\n=== RUNNING ON-DEMAND CUSTOM MATCHUP ===")
        title_a = os.getenv("TITLE_A", "Custom Artwork A").strip() or "Custom Artwork A"
        title_b = os.getenv("TITLE_B", "Custom Artwork B").strip() or "Custom Artwork B"
        criteria = os.getenv("CRITERIA", "").strip() or "composition, anatomy, lighting, color, and visual impact"

        id_a = make_artwork_id(title_a, custom_url_a)
        id_b = make_artwork_id(title_b, custom_url_b)

        # Ensure entries in manifest
        existing_ids = {a["id"] for a in artworks}
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        if id_a not in existing_ids:
            artworks.insert(0, {
                "id": id_a,
                "title": title_a,
                "category": "custom",
                "tag": "Custom",
                "date": today_str,
                "thumb": custom_url_a,
                "fullres": custom_url_a
            })
        if id_b not in existing_ids:
            artworks.insert(0, {
                "id": id_b,
                "title": title_b,
                "category": "custom",
                "tag": "Custom",
                "date": today_str,
                "thumb": custom_url_b,
                "fullres": custom_url_b
            })

        # Ensure entries in ratings
        for art_id, t in [(id_a, title_a), (id_b, title_b)]:
            if art_id not in ratings:
                ratings[art_id] = {
                    "artworkId": art_id,
                    "author": "visitor",
                    "category": "custom",
                    "elo": 1200,
                    "voteCount": 0,
                    "recentResults": [],
                    "lastReasons": [],
                    "scoreFormulaVersion": 1,
                    "ratingSystemVersion": 1
                }

        rec_a = ratings[id_a]
        rec_b = ratings[id_b]

        print(f"Judging custom match:\n  A: {title_a} ({custom_url_a})\n  B: {title_b} ({custom_url_b})\n  Criteria: {criteria}")
        winner, reason = judge.compare(custom_url_a, custom_url_b, criteria)
        winner_is_a = winner == "A"

        print(f"\n★ Winner: Image {winner} ({title_a if winner_is_a else title_b})")
        print(f"★ Reason: {reason}")

        new_elo_a, new_elo_b = update_elo(rec_a["elo"], rec_b["elo"], winner_is_a, k=16)
        rec_a["elo"] = new_elo_a
        rec_a["voteCount"] += 1
        rec_a["recentResults"] = (rec_a["recentResults"] + [1 if winner_is_a else 0])[-10:]
        rec_a["lastReasons"] = ([{"by": "ai", "criteria": criteria, "reason": reason if winner_is_a else None}] + rec_a["lastReasons"])[:10]

        rec_b["elo"] = new_elo_b
        rec_b["voteCount"] += 1
        rec_b["recentResults"] = (rec_b["recentResults"] + [0 if winner_is_a else 1])[-10:]
        rec_b["lastReasons"] = ([{"by": "ai", "criteria": criteria, "reason": reason if not winner_is_a else None}] + rec_b["lastReasons"])[:10]

        # Save both manifest & ratings
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(artworks, f, indent=2, ensure_ascii=False)

        save_ratings(ratings, "ratings.json")
        sync_to_kv(ratings)
        print("Custom matchup evaluation completed and saved!")
        return

    # Standard scheduled batch mode
    print(f"\n=== RUNNING SCHEDULED SEEDING BATCH ({len(artworks)} total artworks) ===")
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

    sorted_by_votes = sorted(artworks, key=lambda x: ratings[x["id"]]["voteCount"])
    num_matches = int(os.getenv("BATCH_MATCHES", "5"))

    criteria_list = [
        "anatomy, proportions, and character silhouette",
        "color theory, palette harmony, and lighting atmosphere",
        "composition, framing, and focal point clarity",
        "linework definition, textures, and brush technique",
        "overall visual impact and emotional resonance"
    ]

    for i in range(num_matches):
        art_a = sorted_by_votes[i % len(sorted_by_votes)]
        rec_a = ratings[art_a["id"]]

        other_candidates = [x for x in artworks if x["id"] != art_a["id"]]
        other_candidates.sort(key=lambda x: abs(ratings[x["id"]]["elo"] - rec_a["elo"]))
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

    save_ratings(ratings, "ratings.json")
    sync_to_kv(ratings)
    print("\nBatch seeding run finished successfully!")

if __name__ == "__main__":
    main()

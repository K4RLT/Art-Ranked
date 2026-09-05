# Art-Ranked Scheduled AI Seeder

Automated scheduled AI-seeding job for [Yiran Studio](https://yiran-studio.vercel.app/art-ranked) as defined in Section 8 of the implementation spec.

Runs **Moondream2** (`vikhyatk/moondream2`) on CPU using GitHub Actions to continuously rate artworks, update Elo ratings ($K=16$), and keep the leaderboard active.

## How it works
1. Runs every 6 hours automatically via GitHub Actions cron schedule (`0 */6 * * *`) or manually via **Run workflow**.
2. Reads [`artworks-manifest.json`](./artworks-manifest.json).
3. Selects pairings prioritizing pieces with lowest vote count (Swiss-style proximity matching).
4. Uses **Moondream2** vision model to visually evaluate both images on criteria (anatomy, composition, color, lighting, impact).
5. Updates Elo and commits the updated [`ratings.json`](./ratings.json) (and syncs to Cloudflare / Vercel KV if secrets are set).

# Deploying the API

The image contains only the API, the trained model (9 KB) and the demo page: no raw data, no research
dependencies. It runs as a non-root user and has a Docker health check.

## Local

```bash
docker compose up --build          # http://127.0.0.1:8080  (demo page at /, API docs at /docs)
```

## DigitalOcean droplet with HTTPS

1. Create an Ubuntu droplet (the smallest 1 GB plan is enough: the model scores ~40k clients per second) and point a
   DNS `A` record, e.g. `uplift.example.com`, at its IP.
2. On the droplet, install Docker and clone the repository:

   ```bash
   curl -fsSL https://get.docker.com | sh
   git clone <your repo url> uplift && cd uplift
   ```

3. Start the API behind Caddy, which obtains a Let's Encrypt certificate on first request:

   ```bash
   export DOMAIN=uplift.example.com
   export UPLIFT_API_KEY=$(openssl rand -hex 24)   # keep this; clients send it as X-API-Key
   docker compose -f deploy/docker-compose.prod.yml up -d --build
   ```

4. Smoke test from your machine:

   ```bash
   curl https://uplift.example.com/health
   curl -X POST https://uplift.example.com/score/batch -H "X-API-Key: $UPLIFT_API_KEY" -F file=@examples/clients_example.csv
   ```

Only ports 80 and 443 need to be open (`ufw allow 80,443/tcp`). `/health`, `/docs`, `/demo/sample` and the demo
page are public; `/score`, `/plan`, `/score/batch` and `/drift` require the key. To make the demo usable by visitors
without a key, leave `UPLIFT_API_KEY` empty only on a demo instance that serves no real customer data.

## Updating the model

```bash
python scripts/train_final.py        # writes models/uplift_model.pkl and models/model_meta.json
pytest -q                             # includes the training/serving skew test
git commit -am "retrain model" && git push
# on the server:
git pull && docker compose -f deploy/docker-compose.prod.yml up -d --build
```

## Operating it

- **Before each campaign:** `POST /drift` with the clients to be scored. `overall: "drift"` means they no longer look
  like the training population; treat the plan as a hypothesis and keep a larger holdout.
- **Every campaign:** call `/plan` with `holdout_share` ≥ 0.1. The holdout is assigned deterministically per
  `client_id`, so regenerating the plan keeps the same control group. The response states the smallest effect the
  campaign will be able to detect.
- **After the campaign:** compare purchase rates of `contact` vs `holdout`. That is the measured effect of the
  targeted campaign, and the contact/holdout pairs are new randomised training data for the next model.

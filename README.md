# crispy-meme
Experiements with OpenAI Agent SDK

# Setup

## Running litellm server:
```bash
docker rm -f litellm || true

docker run -d \
  --name litellm \
  -p 4000:4000 \
  --env-file .env \
  -v $(pwd)/litellm_config.yml:/app/config.yml:ro \
  ghcr.io/berriai/litellm:main-latest \
  --config /app/config.yml \
  --port 4000 \
  --host 0.0.0.0 
  
docker logs -f litellm
```

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash",
    "messages": [{"role": "user", "content": "Say hello!"}]
  }'
```

## litellm_config
```yaml
model_list:
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.0-flash-exp
      api_key: ${GEMINI_API_KEY}
```

## Debugging Gemini API directly
```shell
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={PASTE_KEY_HERE}" \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Hello!"
      }]
    }]
  }'
```
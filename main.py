# Run the app in background
python3 app.py &

# Wait 3 seconds for it to start
sleep 3

# Test the API directly
curl -s -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "space", "method": "tfidf", "relevant": []}' | python3 -m json.tool

# Stop the background app
kill %1
deactivate

#!/bin/bash
# Bounded resumable fetch: each attempt <=170s, up to MAX_ATTEMPTS, using HTTP Range via curl -C -
URL="$1"; OUT="$2"; EXPECT="$3"; MAX_ATTEMPTS="${4:-12}"
for i in $(seq 1 $MAX_ATTEMPTS); do
  sz=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$sz" -ge "$EXPECT" ]; then echo "DONE size=$sz"; break; fi
  echo "attempt $i start size=$sz $(date +%H:%M:%S)"
  curl -s -C - -m 170 -A "promoter-atlas-research/1.0" -o "$OUT" "$URL" || echo "curl_exit=$?"
  sz2=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "attempt $i end size=$sz2 $(date +%H:%M:%S)"
  if [ "$sz2" -le "$sz" ]; then echo "no progress; stop"; break; fi
done

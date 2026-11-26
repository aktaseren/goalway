#!/usr/bin/env bash

set -e

HOST=$(oc get route fastapi-demo -o jsonpath='{.spec.host}')

echo "Creating conversation..."

RESP1=$(curl -s -X POST http://$HOST/v1/chat/completions \
-H "Content-Type: application/json" \
-H "x-api-key: redhat-demo-key" \
-d '{
  "model":"tinyllama",
  "messages":[
    {"role":"user","content":"My name is Eren. Remember this."}
  ]
}')

echo "$RESP1"

CONV_ID=$(echo "$RESP1" | python3 -c "import sys,json; print(json.load(sys.stdin)['gateway']['conversation_id'])")

echo
echo "Conversation ID: $CONV_ID"
echo
echo "Asking follow-up..."

curl -s -X POST http://$HOST/v1/chat/completions \
-H "Content-Type: application/json" \
-H "x-api-key: redhat-demo-key" \
-d "{
  \"model\":\"tinyllama\",
  \"conversation_id\":\"$CONV_ID\",
  \"messages\":[
    {\"role\":\"user\",\"content\":\"What is my name?\"}
  ]
}"

echo

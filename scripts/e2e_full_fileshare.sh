#!/bin/bash
# Full file-sharing E2E: D (empty model dir) starts model present only on A.
# Requirement: D should import weights from A over loopback, then spin the cluster.
TMP=/media/saumitra-koleshwar/Extra/tmp/opencode
NODE=/media/extra/Symlinks/BitNet/terllama-repo/build-release/terllama-node
pkill -f terllama-node 2>/dev/null || true
pkill -f terllama-worker 2>/dev/null || true
pkill -f terllama-cluster 2>/dev/null || true
sleep 2
rm -rf "$TMP/test-models-D"
mkdir -p "$TMP/test-models-D"
setsid nohup "$NODE" --http-port 47801 --fake-ram 3072 --peer 127.0.0.1:47804 > "$TMP/nodeA.log" 2>&1 < /dev/null &
setsid nohup env TERLLAMA_MODEL_DIR="$TMP/test-models-D" "$NODE" --http-port 47804 --fake-ram 2048 --peer 127.0.0.1:47801 > "$TMP/nodeD.log" 2>&1 < /dev/null &
disown -a
sleep 5
echo "=== D peers ==="
curl -s --max-time 5 http://127.0.0.1:47804/node/peers | python3 -c "import json,sys; [print(p['node_id'][:6], p['host'], p['http_port'], p['ram_available_mb']) for p in json.load(sys.stdin)['peers']]"
echo "=== D models (should be empty) ==="
curl -s --max-time 5 http://127.0.0.1:47804/node/models
echo
echo "=== start-model on D (leader, imports from A) ==="
time curl -s --max-time 240 -X POST http://127.0.0.1:47804/node/start-model -H "Content-Type: application/json" -d '{"model_name":"terllama-convert-test"}' | head -c 800
echo
echo "=== D models after (should list the imported model) ==="
curl -s --max-time 5 http://127.0.0.1:47804/node/models
echo
echo "=== D running clusters ==="
curl -s --max-time 5 http://127.0.0.1:47804/node/status | python3 -c "import json,sys; d=json.load(sys.stdin); [print(rc['model'], rc['cluster_port'], len(rc['workers']), 'workers') for rc in d['running']]" 2>/dev/null
echo "=== processes ==="
pgrep -af 'terllama-worker|terllama-node' | head -12
/*
 * node.h — terllama-node: LAN discovery, model management, cluster spawner.
 *
 * Standalone daemon that:
 *   - announces itself on UDP multicast 239.253.1.1:47800 and keeps a live
 *     peer table (beacon every 2s, peers expire after 8s),
 *   - serves a JSON HTTP API (/node/...) for model + cluster management,
 *   - acts as a leader that shards a model across the nodes it sees and
 *     spawns one terllama-worker per participant plus one terllama-cluster.
 *
 * File-sharing design (used by POST /node/start-model, step 5):
 *   "import-model pull" — the leader never store-and-forwards file bytes.
 *   A node that needs a model it lacks calls POST /node/import-model with a
 *   `from` URL; the target then pulls every file via GET /node/files/{model}/{file}
 *   binary streams from the source and verifies sizes. The leader first makes
 *   sure IT has a local copy (importing from the first node that had the
 *   model), then asks each participating peer to import from the leader.
 *
 * Shard plan: participants = this node (rank 0) + every live peer sorted by
 * node_id. RAM (MB) is converted to bytes and fed to tldist::compute_shards
 * (partitioner.h) together with the model's total size; the partitioner
 * returns contiguous half-open [start,end) ranges in pipeline rank order.
 *
 * Port allocation:
 *   - worker ports: scanned per participant host from 47900 upward; a port is
 *     free when GET http://<host>:<port>/health produces no HTTP response.
 *   - cluster port: scanned on the leader from 48000 upward (bind probe),
 *     then waited on until GET /v1/models returns 200.
 */
#pragma once

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <httplib.h>
#include <json.hpp>

namespace tlnode {

// ─── Discovery constants ───────────────────────────────────────────────────
constexpr uint16_t kBeaconPort = 47800;            // UDP multicast port
constexpr uint32_t kBeaconMagic = 0x544C524Eu;     // "TLRN"
constexpr int kBeaconSize = 128;                   // fixed beacon payload
constexpr int64_t kBeaconIntervalMs = 2000;
constexpr int64_t kPeerExpiryMs = 8000;

// ─── Cluster / port allocation ─────────────────────────────────────────────
constexpr int kWorkerPortBase = 47900;
constexpr int kClusterPortBase = 48000;
constexpr int kMaxRunningClusters = 3;

// ─── Data structures ───────────────────────────────────────────────────────
struct PeerInfo {
    uint64_t node_id = 0;
    std::string name;
    std::string host;          // IP the beacon came from / --peer host
    int http_port = 0;
    uint32_t ram_available_mb = 0;
    int64_t last_seen_ms = 0;  // steady_clock ms
};

struct ModelFileInfo {
    std::string name;
    int64_t size = 0;
    std::string sha256;
};

struct WorkerInfo {
    uint64_t node_id = 0;
    std::string node_name;
    std::string host;
    int port = 0;
    int start = 0;
    int end = 0;
};

struct RunningCluster {
    std::string model;
    int cluster_port = 0;
    pid_t cluster_pid = 0;      // leader side only
    std::vector<WorkerInfo> workers;
};

struct ChildProc {
    pid_t pid = 0;
    std::string tag;
};

struct NodeOptions {
    int http_port = 47801;
    uint32_t fake_ram_mb = 0;              // 0 = use sysinfo()
    uint64_t forced_node_id = 0;           // 0 = load/persist ~/.terllama/node_id
    std::vector<std::pair<std::string, int>> peers;  // unicast beacon targets
};

// ─── Node daemon ───────────────────────────────────────────────────────────
class Node {
public:
    explicit Node(const NodeOptions& opts);
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // Starts the discovery thread and the HTTP server; blocks until stop().
    void run();

    // Kills all children, stops the server and the discovery thread.
    void stop();

private:
    // ── Discovery ──────────────────────────────────────────────────────
    void discovery_loop();
    void send_beacon(int fd);

    // ── HTTP ───────────────────────────────────────────────────────────
    void register_routes();
    void handle_status(const httplib::Request& req, httplib::Response& res);
    void handle_start_model(const httplib::Request& req, httplib::Response& res);
    void handle_stop_model(const httplib::Request& req, httplib::Response& res);
    void handle_start_worker(const httplib::Request& req, httplib::Response& res);
    void handle_stop_worker(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_files_list(const httplib::Request& req, httplib::Response& res);
    void handle_file_stream(const httplib::Request& req, httplib::Response& res);
    void handle_pull_model(const httplib::Request& req, httplib::Response& res);
    void handle_import_model(const httplib::Request& req, httplib::Response& res);

    // ── Helpers ────────────────────────────────────────────────────────
    std::vector<PeerInfo> live_peers();
    int peer_http_port(uint64_t node_id);
    bool peer_has_model(const PeerInfo& p, const std::string& model);
    int peer_running_count(const PeerInfo& p);
    bool wait_http_ok(const std::string& host, int port,
                      const std::string& path, int timeout_s);
    bool port_responds(const std::string& host, int port);
    int find_free_worker_port(const std::string& host);
    int find_free_cluster_port();
    bool import_model_from(const std::string& from_host, int from_port,
                           const std::string& model, std::string& err);
    void stop_local_worker(int port);
    void cleanup_partial_cluster(const std::vector<WorkerInfo>& workers, pid_t cluster_pid);
    void shutdown_children();

    // ── State ──────────────────────────────────────────────────────────
    NodeOptions opts_;
    std::atomic<bool> stop_{false};

    uint64_t node_id_ = 0;
    std::string name_;
    uint32_t ram_mb_ = 0;
    std::string exe_dir_;

    httplib::Server svr_;
    std::thread discovery_th_;

    std::mutex peers_mu_;
    std::map<uint64_t, PeerInfo> peers_;

    std::mutex running_mu_;
    std::vector<RunningCluster> running_;

    std::mutex workers_mu_;
    std::map<int, pid_t> worker_pids_;     // port → local worker pid

    std::mutex children_mu_;
    std::vector<ChildProc> children_;
};

// ─── Free helpers (shared with node_main.cpp) ──────────────────────────────
// Resolves "<exe_dir>/<name>" if present, else the bare name (PATH lookup).
std::string resolve_exe(const std::string& name);

// Forks + execvp. Children inherit stdout/stderr. Returns pid or -1.
pid_t spawn_process(const std::vector<std::string>& argv);

// SIGTERM, wait up to 5s, then SIGKILL + reap.
void kill_child(pid_t pid, const std::string& tag);

// Models dir: $TERLLAMA_MODEL_DIR else ~/.terllama/models.
std::string models_dir();

// Loads the persisted node id (creating ~/.terllama/node_id on first run).
uint64_t load_or_create_node_id();

// Hex SHA-256 of a file; empty string on read failure.
std::string sha256_hex(const std::string& path);

}  // namespace tlnode

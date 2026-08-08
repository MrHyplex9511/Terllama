/*
 * node.cpp — terllama-node implementation.
 *
 * See node.h for the design overview (file-sharing, shard plan, ports).
 */
#include "node.h"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "core/logger.h"
#include "partitioner.h"

namespace tlnode {

namespace {

namespace fs = std::filesystem;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ─── Fixed beacon payload (first 82 bytes of the 128-byte datagram) ────────
#pragma pack(push, 1)
struct Beacon {
    uint32_t magic;            // kBeaconMagic, net order
    uint64_t node_id;          // net order
    uint16_t http_port;        // net order
    uint32_t ram_available_mb; // net order
    char name[64];             // null-padded
};
#pragma pack(pop)

static_assert(sizeof(Beacon) <= (size_t)kBeaconSize, "beacon too large");

inline bool is_little_endian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

uint64_t bswap64(uint64_t v) {
    return __builtin_bswap64(v);
}

uint64_t htonll(uint64_t v) { return is_little_endian() ? bswap64(v) : v; }
uint64_t ntohll(uint64_t v) { return htonll(v); }

std::string hex_id(uint64_t id) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << id;
    return os.str();
}

// ─── Minimal SHA-256 (FIPS 180-4), no external deps ────────────────────────
constexpr uint32_t kShaK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

std::string sha256_file(const std::string& path) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t total = 0;
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";

    auto process_block = [&h](const uint8_t* p) {
        uint32_t w[64];
        for (int t = 0; t < 16; t++)
            w[t] = ((uint32_t)p[t * 4] << 24) | ((uint32_t)p[t * 4 + 1] << 16) |
                   ((uint32_t)p[t * 4 + 2] << 8) | (uint32_t)p[t * 4 + 3];
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = rotr32(w[t - 15], 7) ^ rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3);
            uint32_t s1 = rotr32(w[t - 2], 17) ^ rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], ff = h[5], g = h[6], hh = h[7];
        for (int t = 0; t < 64; t++) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & ff) ^ ((~e) & g);
            uint32_t t1 = hh + S1 + ch + kShaK[t] + w[t];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = ff; ff = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += ff; h[6] += g; h[7] += hh;
    };

    std::vector<uint8_t> carry;
    std::array<uint8_t, 65536> buf;
    while (true) {
        f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size());
        auto n = (size_t)f.gcount();
        if (n == 0) break;
        total += n;
        carry.insert(carry.end(), buf.begin(), buf.begin() + (std::ptrdiff_t)n);
        size_t i = 0;
        for (; i + 64 <= carry.size(); i += 64) process_block(carry.data() + i);
        carry.erase(carry.begin(), carry.begin() + (std::ptrdiff_t)i);
        if (n < buf.size()) break;  // EOF
    }

    uint64_t bitlen = total * 8;
    carry.push_back(0x80);
    while (carry.size() % 64 != 56) carry.push_back(0);
    for (int t = 7; t >= 0; t--) carry.push_back((uint8_t)(bitlen >> (t * 8)));
    for (size_t i = 0; i + 64 <= carry.size(); i += 64) process_block(carry.data() + i);

    static const char hexc[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int t = 0; t < 8; t++)
        for (int s = 28; s >= 0; s -= 4) out += hexc[(h[t] >> s) & 0xF];
    return out;
}

// Runs argv[0..] with stdout+stderr captured; returns {exit_code, output}.
std::pair<int, std::string> run_capture(const std::vector<std::string>& argv) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return {-1, ""};
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return {-1, ""}; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        std::vector<char*> cstr;
        for (auto& a : argv) cstr.push_back(const_cast<char*>(a.c_str()));
        cstr.push_back(nullptr);
        execvp(cstr[0], cstr.data());
        _exit(127);
    }
    close(pipefd[1]);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) out.append(buf, (size_t)n);
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
    return {code, out};
}

// ─── Model dir helpers ─────────────────────────────────────────────────────
bool valid_model_name(const std::string& name) {
    if (name.empty() || name == "." || name == ".." || name[0] == '.') return false;
    return name.find('/') == std::string::npos;
}

std::string model_dir_of(const std::string& name) {
    return models_dir() + "/" + name;
}

bool dir_has_model(const std::string& dir) {
    return fs::exists(dir + "/model_extra.bin");
}

// File-list + sha256 cache. Hashing a multi-hundred-MB model on every
// /node/status or /node/models poll cost seconds on cold cache (and repeated
// statfs work regardless). Model files are immutable after conversion, so we
// fingerprint each dir with cheap (name,size,mtime) stats and only re-hash
// when the fingerprint changes.
struct ModelDirCache {
    std::vector<ModelFileInfo> files;
    std::string fingerprint;  // "name:size:mtime_ns;" per file
};
std::map<std::string, ModelDirCache> g_model_cache;
std::mutex g_model_cache_mu;

std::string dir_fingerprint(const std::string& dir) {
    std::string fp;
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        auto name = it->path().filename().string();
        if (name.empty() || name[0] == '.') continue;
        auto mtime = it->last_write_time(ec);
        fp += name + ":" + std::to_string(it->file_size(ec)) + ":" +
              std::to_string(mtime.time_since_epoch().count()) + ";";
    }
    return fp;
}

std::vector<ModelFileInfo> list_model_files(const std::string& dir) {
    std::string fp = dir_fingerprint(dir);
    {
        std::lock_guard<std::mutex> lk(g_model_cache_mu);
        auto it = g_model_cache.find(dir);
        if (it != g_model_cache.end() && it->second.fingerprint == fp)
            return it->second.files;
    }
    std::vector<ModelFileInfo> out;
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        auto name = it->path().filename().string();
        if (name.empty() || name[0] == '.') continue;
        ModelFileInfo m;
        m.name = name;
        m.size = (int64_t)it->file_size();
        m.sha256 = sha256_file(it->path().string());
        out.push_back(std::move(m));
    }
    std::sort(out.begin(), out.end(),
              [](const ModelFileInfo& a, const ModelFileInfo& b) { return a.name < b.name; });
    {
        std::lock_guard<std::mutex> lk(g_model_cache_mu);
        g_model_cache[dir] = ModelDirCache{out, std::move(fp)};
    }
    return out;
}

int64_t dir_total_size(const std::string& dir) {
    int64_t total = 0;
    for (auto& m : list_model_files(dir)) total += m.size;
    return total;
}

bool is_local_ipv4(const struct in_addr& addr) {
    uint32_t target = ntohl(addr.s_addr);
    ifaddrs* ifa = nullptr;
    if (getifaddrs(&ifa) != 0 || ifa == nullptr) return false;
    bool found = false;
    for (auto* p = ifa; p != nullptr; p = p->ifa_next) {
        if (p->ifa_addr == nullptr || p->ifa_addr->sa_family != AF_INET) continue;
        auto* sin = reinterpret_cast<struct sockaddr_in*>(p->ifa_addr);
        if (ntohl(sin->sin_addr.s_addr) == target) {
            found = true;
            break;
        }
    }
    freeifaddrs(ifa);
    return found;
}

std::string local_ip() {
    ifaddrs* ifa = nullptr;
    if (getifaddrs(&ifa) != 0 || ifa == nullptr) return "127.0.0.1";
    std::string ip = "127.0.0.1";
    for (auto* p = ifa; p != nullptr; p = p->ifa_next) {
        if (p->ifa_addr == nullptr || p->ifa_addr->sa_family != AF_INET) continue;
        if (p->ifa_flags & IFF_LOOPBACK) continue;
        char buf[INET_ADDRSTRLEN];
        auto* sin = reinterpret_cast<struct sockaddr_in*>(p->ifa_addr);
        if (inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) {
            ip = buf;
            break;
        }
    }
    freeifaddrs(ifa);
    return ip;
}

}  // namespace

// ─── Free helpers ──────────────────────────────────────────────────────────
std::string models_dir() {
    const char* env = std::getenv("TERLLAMA_MODEL_DIR");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/root") + "/.terllama/models";
}

std::string resolve_exe(const std::string& name) {
    std::string self;
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        self = buf;
        size_t slash = self.rfind('/');
        if (slash != std::string::npos) {
            std::string cand = self.substr(0, slash + 1) + name;
            if (fs::exists(cand)) return cand;
        }
    }
    return name;  // fall back to PATH lookup
}

pid_t spawn_process(const std::vector<std::string>& argv) {
    if (argv.empty()) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        std::vector<char*> cstr;
        for (auto& a : argv) cstr.push_back(const_cast<char*>(a.c_str()));
        cstr.push_back(nullptr);
        execvp(cstr[0], cstr.data());
        _exit(127);
    }
    return pid;
}

void kill_child(pid_t pid, const std::string& tag) {
    if (pid <= 0) return;
    Logger::info("terllama-node: stopping child ({} pid {})", tag, pid);
    ::kill(pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
        int st = 0;
        if (waitpid(pid, &st, WNOHANG) == pid) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

// Node identity is per-PROCESS, not per-host: the daemon owns the workers and
// cluster it spawns, so a restart legitimately yields a new node. Persisting a
// single id under ~/.terllama/node_id made every daemon on one machine share
// the same id, so they filtered out each other's beacons and never discovered
// one another (zero peers in multi-node-per-host tests). Always draw a fresh
// random id; --node-id still overrides it for deterministic tests.
uint64_t load_or_create_node_id() {
    uint64_t id = 0;
    std::ifstream urnd("/dev/urandom", std::ios::binary);
    if (urnd) {
        urnd.read(reinterpret_cast<char*>(&id), sizeof(id));
        if (id == 0) id = 1;
    } else {
        id = (uint64_t)getpid() ^ (uint64_t)time(nullptr) ^ (uint64_t)now_ms();
    }
    return id;
}

std::string sha256_hex(const std::string& path) { return sha256_file(path); }

// ─── Node ──────────────────────────────────────────────────────────────────
Node::Node(const NodeOptions& opts) : opts_(opts) {
    node_id_ = opts.forced_node_id != 0 ? opts.forced_node_id : load_or_create_node_id();
    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) != 0) std::strncpy(host, "unknown", sizeof(host) - 1);
    name_ = host;

    if (opts.fake_ram_mb != 0) {
        ram_mb_ = opts.fake_ram_mb;
    } else {
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            ram_mb_ = (uint32_t)((si.freeram * si.mem_unit) >> 20);
        } else {
            ram_mb_ = 0;
        }
    }
}

Node::~Node() { stop(); }

void Node::stop() {
    stop_.store(true);
    // Stop remote workers of tracked clusters before local teardown so no
    // participant is left with orphaned terllama-worker processes.
    std::vector<RunningCluster> running;
    {
        std::lock_guard<std::mutex> lk(running_mu_);
        running = std::move(running_);
        running_.clear();
    }
    for (auto& rc : running) {
        for (auto& w : rc.workers) {
            if (w.node_id == node_id_) continue;  // killed by shutdown_children()
            int hp = peer_http_port(w.node_id);
            if (hp <= 0) continue;
            httplib::Client cli(w.host, hp);
            cli.set_connection_timeout(1, 0);
            cli.set_read_timeout(2, 0);
            cli.Post("/node/stop-worker", nlohmann::json{{"port", w.port}}.dump(),
                     "application/json");
        }
    }
    shutdown_children();
    svr_.stop();
    if (discovery_th_.joinable()) discovery_th_.join();
}

void Node::shutdown_children() {
    std::lock_guard<std::mutex> lk(children_mu_);
    for (auto& c : children_) {
        if (c.pid > 0) ::kill(c.pid, SIGTERM);
    }
    for (auto& c : children_) {
        if (c.pid <= 0) continue;
        for (int i = 0; i < 50; i++) {
            int st = 0;
            if (waitpid(c.pid, &st, WNOHANG) == c.pid) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ::kill(c.pid, SIGKILL);
        waitpid(c.pid, nullptr, 0);
    }
    children_.clear();
}

// ─── Discovery ─────────────────────────────────────────────────────────────
void Node::discovery_loop() {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        Logger::error("terllama-node: discovery socket failed");
        return;
    }
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in bind_addr {};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port = htons(kBeaconPort);
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0) {
        Logger::error("terllama-node: discovery bind 0.0.0.0:{} failed", kBeaconPort);
        close(fd);
        return;
    }

    struct ip_mreq mreq {};
    mreq.imr_multiaddr.s_addr = inet_addr("239.253.1.1");
    mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0)
        Logger::warn("terllama-node: IP_ADD_MEMBERSHIP failed (multicast disabled)");
    unsigned char ttl = 2;
    setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

    struct timeval tv {0, 300 * 1000};  // 300 ms recv timeout
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    Logger::info("terllama-node: discovery on 239.253.1.1:{} (node_id {})", kBeaconPort,
                 hex_id(node_id_));
    int64_t last_send = 0;
    while (!stop_.load()) {
        char buf[kBeaconSize];
        struct sockaddr_in from {};
        socklen_t fromlen = sizeof(from);
        ssize_t n = recvfrom(fd, buf, sizeof(buf), 0,
                             reinterpret_cast<struct sockaddr*>(&from), &fromlen);
        if (n >= (ssize_t)sizeof(Beacon)) {
            auto* b = reinterpret_cast<const Beacon*>(buf);
            if (ntohl(b->magic) == kBeaconMagic) {
                uint64_t id = ntohll(b->node_id);
                if (id != node_id_) {
                    PeerInfo p;
                    p.node_id = id;
                    p.name = std::string(b->name, strnlen(b->name, sizeof(b->name)));
                    // Peers on this same host must be addressed via loopback, not
                    // via whichever LAN/VPN address the beacon arrived on.
                    p.host = is_local_ipv4(from.sin_addr) ? "127.0.0.1" : inet_ntoa(from.sin_addr);
                    p.http_port = ntohs(b->http_port);
                    p.ram_available_mb = ntohl(b->ram_available_mb);
                    p.last_seen_ms = now_ms();
                    {
                        std::lock_guard<std::mutex> lk(peers_mu_);
                        peers_[id] = std::move(p);
                    }
                    Logger::debug("terllama-node: peer {} ({}:{})", hex_id(id), p.host,
                                  p.http_port);
                }
            }
        }
        if (now_ms() - last_send >= kBeaconIntervalMs) {
            send_beacon(fd);
            last_send = now_ms();
        }
    }
    close(fd);
}

void Node::send_beacon(int fd) {
    char buf[kBeaconSize] = {0};
    auto* b = reinterpret_cast<Beacon*>(buf);
    b->magic = htonl(kBeaconMagic);
    b->node_id = htonll(node_id_);
    b->http_port = htons((uint16_t)opts_.http_port);
    b->ram_available_mb = htonl(ram_mb_);
    std::strncpy(b->name, name_.c_str(), sizeof(b->name) - 1);

    struct sockaddr_in dst {};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(kBeaconPort);
    dst.sin_addr.s_addr = inet_addr("239.253.1.1");
    sendto(fd, buf, kBeaconSize, 0, reinterpret_cast<struct sockaddr*>(&dst), sizeof(dst));

    for (auto& [host, port] : opts_.peers) {
        (void)port;  // peer port is a hint (HTTP port); beacons always land on
                     // the well-known UDP beacon port where every node listens.
        struct sockaddr_in peer {};
        peer.sin_family = AF_INET;
        peer.sin_port = htons(kBeaconPort);
        if (inet_pton(AF_INET, host.c_str(), &peer.sin_addr) != 1) continue;
        sendto(fd, buf, kBeaconSize, 0, reinterpret_cast<struct sockaddr*>(&peer), sizeof(peer));
    }
}

// ─── Peer helpers ──────────────────────────────────────────────────────────
std::vector<PeerInfo> Node::live_peers() {
    std::lock_guard<std::mutex> lk(peers_mu_);
    int64_t cutoff = now_ms() - kPeerExpiryMs;
    std::vector<PeerInfo> out;
    for (auto it = peers_.begin(); it != peers_.end();) {
        if (it->second.last_seen_ms < cutoff) {
            it = peers_.erase(it);
        } else {
            out.push_back(it->second);
            ++it;
        }
    }
    std::sort(out.begin(), out.end(),
              [](const PeerInfo& a, const PeerInfo& b) { return a.node_id < b.node_id; });
    return out;
}

int Node::peer_http_port(uint64_t node_id) {
    std::lock_guard<std::mutex> lk(peers_mu_);
    auto it = peers_.find(node_id);
    return it != peers_.end() ? it->second.http_port : 0;
}

bool Node::peer_has_model(const PeerInfo& p, const std::string& model) {
    httplib::Client cli(p.host, p.http_port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);
    auto r = cli.Get("/node/models");
    if (!r || r->status != 200) return false;
    try {
        auto j = nlohmann::json::parse(r->body);
        for (auto& m : j.value("models", nlohmann::json::array()))
            if (m.value("name", "") == model) return true;
    } catch (...) {}
    return false;
}

int Node::peer_running_count(const PeerInfo& p) {
    httplib::Client cli(p.host, p.http_port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(5, 0);
    auto r = cli.Get("/node/status");
    if (!r || r->status != 200) return 0;
    try {
        return (int)nlohmann::json::parse(r->body).value("running", nlohmann::json::array()).size();
    } catch (...) {
        return 0;
    }
}

bool Node::wait_http_ok(const std::string& host, int port, const std::string& path, int timeout_s) {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        auto r = cli.Get(path);
        if (r && r->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

bool Node::port_responds(const std::string& host, int port) {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(0, 400 * 1000);
    cli.set_read_timeout(0, 400 * 1000);
    auto r = cli.Get("/health");
    return r && r->status == 200;
}

int Node::find_free_worker_port(const std::string& host) {
    for (int p = kWorkerPortBase; p < 65536; p++)
        if (!port_responds(host, p)) return p;
    return -1;
}

int Node::find_free_cluster_port() {
    for (int p = kClusterPortBase; p < 65536; p++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return p;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in a {};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons((uint16_t)p);
        if (bind(fd, reinterpret_cast<struct sockaddr*>(&a), sizeof(a)) == 0) {
            close(fd);
            return p;
        }
        close(fd);
    }
    return -1;
}

// ─── Import model (pull each file via GET /node/files/...) ─────────────────
bool Node::import_model_from(const std::string& from_host, int from_port,
                             const std::string& model, std::string& err) {
    if (!valid_model_name(model)) {
        err = "invalid model name";
        return false;
    }
    httplib::Client src(from_host, from_port);
    src.set_connection_timeout(5, 0);
    src.set_read_timeout(600, 0);

    auto list = src.Get("/node/files/" + model);
    if (!list || list->status != 200) {
        err = "cannot list files at " + from_host + ":" + std::to_string(from_port) + "/node/files/" + model;
        return false;
    }
    std::vector<ModelFileInfo> files;
    try {
        for (auto& f : nlohmann::json::parse(list->body).value("files", nlohmann::json::array()))
            files.push_back({f.value("name", ""), f.value("size", (int64_t)0), ""});
    } catch (const std::exception& e) {
        err = std::string("bad file list: ") + e.what();
        return false;
    }
    if (files.empty()) {
        err = "model has no files";
        return false;
    }

    std::string dir = model_dir_of(model);
    std::error_code ec;
    fs::create_directories(dir, ec);

    for (auto& f : files) {
        if (!valid_model_name(f.name)) {
            err = "refusing unsafe file name " + f.name;
            return false;
        }
        std::string tmp = dir + "/." + f.name + ".part";
        std::string final_path = dir + "/" + f.name;
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            err = "cannot write " + tmp;
            return false;
        }
        int64_t got = 0;
        auto r = src.Get("/node/files/" + model + "/" + f.name,
                         [&](const char* data, size_t n) {
                             out.write(data, (std::streamsize)n);
                             got += (int64_t)n;
                             return true;
                         });
        out.close();
        if (!r || r->status != 200 || got != f.size) {
            fs::remove(tmp, ec);
            err = "download of " + f.name + " failed (got " + std::to_string(got) +
                  " of " + std::to_string(f.size) + " bytes)";
            return false;
        }
        fs::rename(tmp, final_path, ec);
    }
    Logger::info("terllama-node: imported model '{}' from {}:{} ({} files)", model,
                 from_host, from_port, files.size());
    return true;
}

// ─── HTTP ──────────────────────────────────────────────────────────────────
void Node::run() {
    discovery_th_ = std::thread(&Node::discovery_loop, this);
    register_routes();
    svr_.new_task_queue = [] { return new httplib::ThreadPool(8); };
    Logger::info("terllama-node: HTTP API on 0.0.0.0:{} (node {} '{}')", opts_.http_port,
                 hex_id(node_id_), name_);
    svr_.listen("0.0.0.0", opts_.http_port);
    if (discovery_th_.joinable()) discovery_th_.join();
}

void Node::register_routes() {
    svr_.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"},
    });
    svr_.Options(".*", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"ok":true})", "application/json");
    });
    svr_.Get("/node/status", [this](const httplib::Request& req, httplib::Response& res) {
        handle_status(req, res);
    });
    svr_.Get("/node/peers", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json peers = nlohmann::json::array();
        for (auto& p : live_peers()) {
            peers.push_back({{"node_id", hex_id(p.node_id)}, {"name", p.name}, {"host", p.host},
                             {"http_port", p.http_port},
                             {"ram_available_mb", p.ram_available_mb},
                             {"last_seen_ms", p.last_seen_ms}});
        }
        res.set_content(nlohmann::json{{"peers", peers}}.dump(), "application/json");
    });
    svr_.Get("/node/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    svr_.Post("/node/start-model", [this](const httplib::Request& req, httplib::Response& res) {
        handle_start_model(req, res);
    });
    svr_.Post("/node/stop-model", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stop_model(req, res);
    });
    svr_.Post("/node/start-worker", [this](const httplib::Request& req, httplib::Response& res) {
        handle_start_worker(req, res);
    });
    svr_.Post("/node/stop-worker", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stop_worker(req, res);
    });
    svr_.Post("/node/pull-model", [this](const httplib::Request& req, httplib::Response& res) {
        handle_pull_model(req, res);
    });
    svr_.Post("/node/import-model", [this](const httplib::Request& req, httplib::Response& res) {
        handle_import_model(req, res);
    });
    svr_.Get("/node/files/([^/]+)/([^/]+)",
             [this](const httplib::Request& req, httplib::Response& res) {
                 handle_file_stream(req, res);
             });
    svr_.Get("/node/files/([^/]+)", [this](const httplib::Request& req, httplib::Response& res) {
        handle_files_list(req, res);
    });
}

void Node::handle_status(const httplib::Request&, httplib::Response& res) {
    nlohmann::json peers = nlohmann::json::array();
    for (auto& p : live_peers()) {
        peers.push_back({{"node_id", hex_id(p.node_id)}, {"name", p.name}, {"host", p.host},
                         {"http_port", p.http_port}, {"ram_available_mb", p.ram_available_mb},
                         {"last_seen_ms", p.last_seen_ms}});
    }
    nlohmann::json models = nlohmann::json::array();
    std::error_code ec;
    if (fs::is_directory(models_dir(), ec)) {
        for (auto it = fs::directory_iterator(models_dir(), fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_directory()) continue;
            std::string name = it->path().filename().string();
            if (!valid_model_name(name)) continue;
            if (!dir_has_model(it->path().string())) continue;
            models.push_back({{"name", name}, {"size_mb", (int64_t)(dir_total_size(it->path().string()) >> 20)}});
        }
    }
    std::vector<RunningCluster> running;
    {
        std::lock_guard<std::mutex> lk(running_mu_);
        running = running_;
    }
    nlohmann::json running_json = nlohmann::json::array();
    for (auto& rc : running) {
        nlohmann::json workers = nlohmann::json::array();
        for (auto& w : rc.workers)
            workers.push_back({{"node_id", hex_id(w.node_id)}, {"node_name", w.node_name},
                               {"port", w.port}, {"start", w.start}, {"end", w.end}});
        running_json.push_back(
            {{"model", rc.model}, {"cluster_port", rc.cluster_port}, {"workers", workers}});
    }
    nlohmann::json out = {
        {"ok", true},
        {"node_id", hex_id(node_id_)},
        {"name", name_},
        {"http_port", opts_.http_port},
        {"ram_available_mb", ram_mb_},
        {"model_dir", models_dir()},
        {"peers", peers},
        {"models", models},
        {"running", running_json},
    };
    res.set_content(out.dump(), "application/json");
}

void Node::handle_models(const httplib::Request&, httplib::Response& res) {
    nlohmann::json models = nlohmann::json::array();
    std::error_code ec;
    if (fs::is_directory(models_dir(), ec)) {
        for (auto it = fs::directory_iterator(models_dir(), fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::directory_iterator(); it.increment(ec)) {
            if (!it->is_directory()) continue;
            std::string name = it->path().filename().string();
            if (!valid_model_name(name) || !dir_has_model(it->path().string())) continue;
            models.push_back({{"name", name},
                              {"size_mb", (int64_t)(dir_total_size(it->path().string()) >> 20)}});
        }
    }
    res.set_content(nlohmann::json{{"models", models}}.dump(), "application/json");
}

void Node::handle_start_model(const httplib::Request& req, httplib::Response& res) {
    auto fail = [&res](int status, const std::string& msg) {
        res.status = status;
        res.set_content(nlohmann::json{{"error", msg}}.dump(), "application/json");
    };
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        return fail(400, std::string("bad JSON: ") + e.what());
    }
    std::string model = body.value("model_name", "");
    if (!valid_model_name(model)) return fail(400, "invalid model_name");

    // 1. Locate the model locally or on a peer; ensure a local copy exists.
    std::string local_dir = model_dir_of(model);
    std::string src_host;
    int src_port = 0;
    auto peers = live_peers();
    if (!dir_has_model(local_dir)) {
        for (auto& p : peers) {
            if (peer_has_model(p, model)) {
                src_host = p.host;
                src_port = p.http_port;
                break;
            }
        }
        if (src_host.empty()) return fail(400, "model not found on any node");
        std::string err;
        if (!import_model_from(src_host, src_port, model, err))
            return fail(500, "leader import failed: " + err);
    }

    // 2. Layer count from `terllama show <model>` (local copy now exists).
    int n_layers = 0;
    {
        auto [code, out] = run_capture({resolve_exe("terllama"), "show", model});
        if (code != 0) return fail(400, "terllama show failed for " + model);
        size_t pos = out.find("Layers:");
        if (pos != std::string::npos) {
            pos += 7;
            while (pos < out.size() && !std::isdigit((unsigned char)out[pos])) pos++;
            n_layers = std::atoi(out.c_str() + pos);
        }
        if (n_layers <= 0) return fail(400, "could not parse layer count for " + model);
    }
    Logger::info("terllama-node: starting model '{}' ({} layers)", model, n_layers);

    // 3. Global running-cluster budget.
    int total_running = 0;
    {
        std::lock_guard<std::mutex> lk(running_mu_);
        total_running += (int)running_.size();
    }
    for (auto& p : peers) total_running += peer_running_count(p);
    if (total_running >= kMaxRunningClusters)
        return fail(409, "maximum " + std::to_string(kMaxRunningClusters) +
                         " running clusters reached");

    // 4. Participants = self (rank 0) + live peers, RAM in bytes.
    std::vector<PeerInfo> parts;
    parts.push_back(PeerInfo{node_id_, name_, "127.0.0.1", opts_.http_port, ram_mb_, 0});
    for (auto& p : peers) parts.push_back(p);

    std::vector<int64_t> ram_bytes;
    for (auto& p : parts) ram_bytes.push_back((int64_t)p.ram_available_mb * 1024 * 1024);
    int64_t model_bytes = dir_total_size(local_dir);

    std::vector<tldist::ShardSpec> shards;
    try {
        shards = tldist::compute_shards(n_layers, (int)parts.size(), ram_bytes, model_bytes);
    } catch (const std::exception& e) {
        return fail(400, std::string("shard planning failed: ") + e.what());
    }

    // 5. Ensure files on every participating peer (leader already has them).
    std::string leader_ip = local_ip();
    for (size_t i = 1; i < parts.size(); i++) {
        auto& p = parts[i];
        if (peer_has_model(p, model)) continue;
        httplib::Client cli(p.host, p.http_port);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(600, 0);
        nlohmann::json j = {{"model_name", model},
                            {"from", "http://" + leader_ip + ":" + std::to_string(opts_.http_port)}};
        auto r = cli.Post("/node/import-model", j.dump(), "application/json");
        if (!r || r->status != 200)
            return fail(500, "peer " + p.host + " failed to import model: " +
                                 (r ? r->body : "unreachable"));
    }

    // 6. Spawn one worker per participant.
    std::vector<WorkerInfo> workers;
    for (size_t i = 0; i < parts.size(); i++) {
        WorkerInfo w;
        w.node_id = parts[i].node_id;
        w.node_name = parts[i].name;
        w.host = (i == 0) ? "127.0.0.1" : parts[i].host;
        w.start = shards[i].start_layer;
        w.end = shards[i].end_layer;
        w.port = find_free_worker_port(w.host);
        if (w.port < 0) {
            cleanup_partial_cluster(workers, 0);
            return fail(500, "no free worker port on " + w.host);
        }
        if (i == 0) {
            // Local worker spawn.
            pid_t pid = spawn_process({resolve_exe("terllama-worker"), "--listen",
                                       "0.0.0.0:" + std::to_string(w.port), "--model", local_dir,
                                       "--shard", std::to_string(w.start) + "," +
                                                     std::to_string(w.end)});
            if (pid < 0) {
                cleanup_partial_cluster(workers, 0);
                return fail(500, "failed to spawn local worker");
            }
            {
                std::lock_guard<std::mutex> lk(workers_mu_);
                worker_pids_[w.port] = pid;
            }
            {
                std::lock_guard<std::mutex> lk(children_mu_);
                children_.push_back({pid, "worker:" + std::to_string(w.port)});
            }
            if (!wait_http_ok("127.0.0.1", w.port, "/health", 10)) {
                cleanup_partial_cluster(workers, pid);
                return fail(500, "local worker did not become healthy on port " +
                                     std::to_string(w.port));
            }
        } else {
            // Remote worker spawn via HTTP.
            httplib::Client cli(parts[i].host, parts[i].http_port);
            cli.set_connection_timeout(3, 0);
            cli.set_read_timeout(15, 0);
            nlohmann::json j = {{"model", model}, {"start", w.start},
                                {"end", w.end},   {"port", w.port}};
            auto r = cli.Post("/node/start-worker", j.dump(), "application/json");
            if (!r || r->status != 200) {
                cleanup_partial_cluster(workers, 0);
                return fail(500, "peer " + parts[i].host + " start-worker failed: " +
                                     (r ? r->body : "unreachable"));
            }
        }
        workers.push_back(w);
        Logger::info("terllama-node: worker on {} shard [{},{}) port {}", w.node_name,
                     w.start, w.end, w.port);
    }

    // 7. Spawn the cluster on this node.
    int cport = find_free_cluster_port();
    if (cport < 0) {
        cleanup_partial_cluster(workers, 0);
        return fail(500, "no free cluster port");
    }
    std::string workers_csv;
    for (size_t i = 0; i < workers.size(); i++) {
        if (i) workers_csv += ",";
        workers_csv += workers[i].host + ":" + std::to_string(workers[i].port);
    }
    pid_t cpid = spawn_process({resolve_exe("terllama-cluster"), "--workers", workers_csv,
                                "--model", local_dir, "--port", std::to_string(cport)});
    if (cpid < 0) {
        cleanup_partial_cluster(workers, 0);
        return fail(500, "failed to spawn terllama-cluster");
    }
    {
        std::lock_guard<std::mutex> lk(children_mu_);
        children_.push_back({cpid, "cluster:" + std::to_string(cport)});
    }
    if (!wait_http_ok("127.0.0.1", cport, "/v1/models", 30)) {
        cleanup_partial_cluster(workers, cpid);
        return fail(500, "terllama-cluster did not come up on port " + std::to_string(cport));
    }

    // 8-9. Record and respond.
    RunningCluster rc;
    rc.model = model;
    rc.cluster_port = cport;
    rc.cluster_pid = cpid;
    rc.workers = workers;
    {
        std::lock_guard<std::mutex> lk(running_mu_);
        running_.push_back(std::move(rc));
    }
    nlohmann::json wj = nlohmann::json::array();
    for (auto& w : workers)
        wj.push_back({{"node_id", hex_id(w.node_id)}, {"node_name", w.node_name},
                      {"port", w.port}, {"start", w.start}, {"end", w.end}});
    Logger::info("terllama-node: cluster for '{}' up at http://{}:{}", model, leader_ip, cport);
    res.set_content(
        nlohmann::json{{"ok", true},
                       {"cluster_url", "http://" + leader_ip + ":" + std::to_string(cport)},
                       {"port", cport}, {"model", model}, {"workers", wj}}
            .dump(),
        "application/json");
}

void Node::cleanup_partial_cluster(const std::vector<WorkerInfo>& workers, pid_t cluster_pid) {
    if (cluster_pid > 0) {
        ::kill(cluster_pid, SIGTERM);
        waitpid(cluster_pid, nullptr, 0);
    }
    for (auto& w : workers) {
        if (w.node_id == node_id_) {
            stop_local_worker(w.port);
        } else {
            int hp = peer_http_port(w.node_id);
            if (hp <= 0) continue;
            httplib::Client cli(w.host, hp);
            cli.set_connection_timeout(2, 0);
            cli.set_read_timeout(5, 0);
            cli.Post("/node/stop-worker", nlohmann::json{{"port", w.port}}.dump(),
                     "application/json");
        }
    }
}

void Node::handle_stop_model(const httplib::Request& req, httplib::Response& res) {
    RunningCluster rc;
    bool found = false;
    try {
        int cport = nlohmann::json::parse(req.body).value("cluster_port", 0);
        std::lock_guard<std::mutex> lk(running_mu_);
        for (auto it = running_.begin(); it != running_.end(); ++it) {
            if (it->cluster_port == cport) {
                rc = std::move(*it);
                running_.erase(it);
                found = true;
                break;
            }
        }
    } catch (...) {}
    if (!found) {
        res.status = 404;
        res.set_content(R"({"error":"no such cluster"})", "application/json");
        return;
    }
    kill_child(rc.cluster_pid, "cluster:" + std::to_string(rc.cluster_port));
    for (auto& w : rc.workers) {
        if (w.node_id == node_id_) {
            stop_local_worker(w.port);
        } else {
            int hp = peer_http_port(w.node_id);
            if (hp <= 0) continue;
            httplib::Client cli(w.host, hp);
            cli.set_connection_timeout(2, 0);
            cli.set_read_timeout(5, 0);
            cli.Post("/node/stop-worker", nlohmann::json{{"port", w.port}}.dump(),
                     "application/json");
        }
    }
    res.set_content(R"({"ok":true})", "application/json");
}

void Node::handle_start_worker(const httplib::Request& req, httplib::Response& res) {
    auto fail = [&res](int status, const std::string& msg) {
        res.status = status;
        res.set_content(nlohmann::json{{"error", msg}}.dump(), "application/json");
    };
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        return fail(400, std::string("bad JSON: ") + e.what());
    }
    std::string model = body.value("model", "");
    int start = body.value("start", 0);
    int end = body.value("end", 0);
    int port = body.value("port", 0);
    if (!valid_model_name(model) || start < 0 || end <= start || port <= 0) {
        return fail(400, "invalid start-worker payload");
    }
    std::string dir = model_dir_of(model);
    if (!dir_has_model(dir)) return fail(400, "model not present on this node");

    pid_t pid = spawn_process({resolve_exe("terllama-worker"), "--listen",
                               "0.0.0.0:" + std::to_string(port), "--model", dir, "--shard",
                               std::to_string(start) + "," + std::to_string(end)});
    if (pid < 0) return fail(500, "failed to spawn worker");
    {
        std::lock_guard<std::mutex> lk(workers_mu_);
        worker_pids_[port] = pid;
    }
    {
        std::lock_guard<std::mutex> lk(children_mu_);
        children_.push_back({pid, "worker:" + std::to_string(port)});
    }
    if (!wait_http_ok("127.0.0.1", port, "/health", 10)) {
        kill_child(pid, "worker:" + std::to_string(port));
        {
            std::lock_guard<std::mutex> lk(workers_mu_);
            worker_pids_.erase(port);
        }
        return fail(500, "worker did not become healthy on port " + std::to_string(port));
    }
    Logger::info("terllama-node: local worker pid {} shard [{},{}) on :{}", pid, start, end,
                 port);
    res.set_content(nlohmann::json{{"ok", true}, {"pid", (int64_t)pid}}.dump(),
                    "application/json");
}

void Node::handle_stop_worker(const httplib::Request& req, httplib::Response& res) {
    int port = 0;
    try {
        port = nlohmann::json::parse(req.body).value("port", 0);
    } catch (...) {}
    if (port > 0) stop_local_worker(port);
    res.set_content(R"({"ok":true})", "application/json");
}

void Node::stop_local_worker(int port) {
    pid_t pid = 0;
    {
        std::lock_guard<std::mutex> lk(workers_mu_);
        auto it = worker_pids_.find(port);
        if (it != worker_pids_.end()) {
            pid = it->second;
            worker_pids_.erase(it);
        }
    }
    if (pid > 0) kill_child(pid, "worker:" + std::to_string(port));
}

void Node::handle_files_list(const httplib::Request& req, httplib::Response& res) {
    std::string model = req.matches.size() > 1 ? req.matches[1].str() : "";
    if (!valid_model_name(model)) {
        res.status = 400;
        res.set_content(R"({"error":"invalid model"})", "application/json");
        return;
    }
    std::string dir = model_dir_of(model);
    if (!dir_has_model(dir)) {
        res.status = 404;
        res.set_content(R"({"error":"model not found"})", "application/json");
        return;
    }
    nlohmann::json files = nlohmann::json::array();
    for (auto& f : list_model_files(dir))
        files.push_back({{"name", f.name}, {"size", f.size}, {"sha256", f.sha256}});
    res.set_content(nlohmann::json{{"files", files}}.dump(), "application/json");
}

void Node::handle_file_stream(const httplib::Request& req, httplib::Response& res) {
    std::string model = req.matches.size() > 1 ? req.matches[1].str() : "";
    std::string file = req.matches.size() > 2 ? req.matches[2].str() : "";
    if (!valid_model_name(model) || !valid_model_name(file)) {
        res.status = 400;
        res.set_content(R"({"error":"invalid path"})", "application/json");
        return;
    }
    std::string path = model_dir_of(model) + "/" + file;
    std::error_code ec;
    auto size = fs::file_size(path, ec);
    if (ec) {
        res.status = 404;
        res.set_content(R"({"error":"file not found"})", "application/json");
        return;
    }
    res.set_content_provider(
        (size_t)size, "application/octet-stream",
        [path](size_t offset, size_t length, httplib::DataSink& sink) -> bool {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;
            f.seekg((std::streamoff)offset);
            if (!f.good()) return false;
            char buf[65536];
            while (length > 0) {
                size_t to_read = std::min(sizeof(buf), length);
                f.read(buf, (std::streamsize)to_read);
                size_t n = (size_t)f.gcount();
                if (n == 0) break;
                if (!sink.write(buf, n)) return false;
                length -= n;
            }
            return true;
        });
}

void Node::handle_pull_model(const httplib::Request& req, httplib::Response& res) {
    std::string model;
    try {
        model = nlohmann::json::parse(req.body).value("model_name", "");
    } catch (...) {}
    if (!valid_model_name(model)) {
        res.status = 400;
        res.set_content(R"({"error":"invalid model_name"})", "application/json");
        return;
    }
    auto log_out = [&](const std::string& out) {
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line))
            if (!line.empty()) Logger::info("  {}", line);
    };
    auto [c1, o1] = run_capture({resolve_exe("terllama"), "pull", model});
    log_out(o1);
    if (c1 != 0) {
        res.status = 500;
        res.set_content(nlohmann::json{{"error", "terllama pull failed (exit " +
                                                     std::to_string(c1) + ")"}}
                            .dump(),
                        "application/json");
        return;
    }
    auto [c2, o2] = run_capture({resolve_exe("terllama"), "convert", "--model", model});
    log_out(o2);
    if (c2 != 0) {
        res.status = 500;
        res.set_content(nlohmann::json{{"error", "terllama convert failed (exit " +
                                                     std::to_string(c2) + ")"}}
                            .dump(),
                        "application/json");
        return;
    }
    Logger::info("terllama-node: pulled and converted '{}'", model);
    res.set_content(R"({"ok":true})", "application/json");
}

void Node::handle_import_model(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(nlohmann::json{{"error", std::string("bad JSON: ") + e.what()}}.dump(),
                        "application/json");
        return;
    }
    std::string model = body.value("model_name", "");
    std::string from = body.value("from", "");
    if (!valid_model_name(model) || from.empty()) {
        res.status = 400;
        res.set_content(R"({"error":"model_name and from are required"})", "application/json");
        return;
    }
    size_t scheme = from.find("://");
    std::string hp = (scheme == std::string::npos) ? from : from.substr(scheme + 3);
    size_t colon = hp.rfind(':');
    if (colon == std::string::npos) {
        res.status = 400;
        res.set_content(R"({"error":"from must be http://host:port"})", "application/json");
        return;
    }
    std::string host = hp.substr(0, colon);
    int port = std::atoi(hp.c_str() + colon + 1);
    if (host.empty() || port <= 0) {
        res.status = 400;
        res.set_content(R"({"error":"invalid from URL"})", "application/json");
        return;
    }
    std::string err;
    if (!import_model_from(host, port, model, err)) {
        res.status = 500;
        res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
        return;
    }
    res.set_content(R"({"ok":true})", "application/json");
}

}  // namespace tlnode

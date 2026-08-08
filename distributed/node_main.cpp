/*
 * node_main.cpp — terllama-node CLI entry point.
 *
 * Usage:
 *   terllama-node [--http-port N] [--peer HOST:PORT]... [--fake-ram MB]
 *
 * Examples:
 *   terllama-node                                    # discover + serve
 *   terllama-node --peer 10.0.0.2:47801              # also beacon a specific node
 *   terllama-node --fake-ram 4096 --node-id 0xbeef   # loopback testing rig
 */
#include "node.h"

#include "core/logger.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

tlnode::Node* g_node = nullptr;

void on_signal(int sig) {
    Logger::info("terllama-node: {} received, shutting down",
                 sig == SIGINT ? "SIGINT" : "SIGTERM");
    if (g_node) g_node->stop();
}

void usage(FILE* out) {
    fprintf(out,
        "usage: terllama-node [--http-port N] [--peer HOST:PORT]... [--fake-ram MB]\n"
        "\n"
        "  --http-port N    HTTP API port (default 47801)\n"
        "  --peer H:P       also unicast beacons to HOST:PORT (repeatable;\n"
        "                   loopback testing of multiple nodes on one host)\n"
        "  --fake-ram MB    override reported available RAM (uneven-RAM testing)\n"
        "  --node-id HEX    override the persisted node id (testing only)\n"
        "  -h, --help       show this help\n");
}

bool parse_host_port(const std::string& s, std::string& host, int& port) {
    size_t colon = s.rfind(':');
    if (colon == std::string::npos) return false;
    host = s.substr(0, colon);
    port = std::atoi(s.c_str() + colon + 1);
    return !host.empty() && port > 0 && port <= 65535;
}

}  // namespace

int main(int argc, char** argv) {
    tlnode::NodeOptions opts;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--http-port" && i + 1 < argc) {
            opts.http_port = std::atoi(argv[++i]);
        } else if (a == "--peer" && i + 1 < argc) {
            std::string host;
            int port = 0;
            if (!parse_host_port(argv[++i], host, port)) {
                fprintf(stderr, "terllama-node: bad --peer '%s' (want HOST:PORT)\n", argv[i]);
                return 1;
            }
            opts.peers.emplace_back(host, port);
        } else if (a == "--fake-ram" && i + 1 < argc) {
            opts.fake_ram_mb = (uint32_t)std::atoi(argv[++i]);
        } else if (a == "--node-id" && i + 1 < argc) {
            opts.forced_node_id = std::strtoull(argv[++i], nullptr, 0);
        } else if (a == "-h" || a == "--help") {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "terllama-node: unknown argument '%s'\n", a.c_str());
            usage(stderr);
            return 1;
        }
    }
    if (opts.http_port <= 0 || opts.http_port > 65535) {
        fprintf(stderr, "terllama-node: invalid --http-port %d\n", opts.http_port);
        return 1;
    }

    tlnode::Node node(opts);
    g_node = &node;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    node.run();
    return 0;
}

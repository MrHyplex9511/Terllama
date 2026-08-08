/*
 * coordinator_main.cpp — terllama-cluster entry point.
 *
 * Usage:
 *   terllama-cluster --port 8375 --workers host:port,host:port --model PATH
 *
 * Example:
 *   terllama-cluster --workers 10.0.0.1:9100,10.0.0.2:9100 --model ~/.terllama/models/smollm2
 */
#include "coordinator.h"

#include <csignal>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --workers host:port[,host:port...] --model PATH [--port N]\n"
              << "  --workers  comma-separated worker endpoints (required)\n"
              << "  --model    path to the Terllama model dir or .gguf (required)\n"
              << "  --port     OpenAI API port (default 8375)\n";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace.
        size_t a = item.find_first_not_of(" \t\r\n");
        size_t b = item.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) item = item.substr(a, b - a + 1);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string workers_arg, model_path, port = "8375";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--workers" && i + 1 < argc) workers_arg = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = argv[++i];
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown argument: " << a << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (workers_arg.empty() || model_path.empty()) {
        std::cerr << "terllama-cluster: --workers and --model are required\n";
        print_usage(argv[0]);
        return 2;
    }

    std::vector<std::string> worker_urls = split_csv(workers_arg);
    if (worker_urls.empty()) {
        std::cerr << "terllama-cluster: no workers parsed from --workers\n";
        return 2;
    }

    try {
        Coordinator coord(worker_urls, model_path, port);
        if (!coord.start()) {
            std::cerr << "terllama-cluster: coordinator failed to start\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "terllama-cluster: " << e.what() << "\n";
        return 1;
    }

    std::cout << "cluster stopped" << std::endl;
    return 0;
}

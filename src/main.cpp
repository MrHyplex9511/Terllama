/*
 * main.cpp — Terllama CLI Entry Point
 *
 * Home of helper functions (home_dir, models_dir, etc.) and the
 * subcommand dispatcher. All command logic lives in src/cli/commands.cpp.
 *
 * Subcommands:
 *   terllama "prompt" [max_tokens] [temp]   ← legacy mode
 *   terllama list                            ← list local models
 *   terllama show <model>                    ← model info
 *   terllama pull <hf-repo> [--format i2s]  ← download from HF
 *   terllama rm <model>                      ← remove a model
 *   terllama serve [--port N]                ← start API server
 *   terllama chat --model <m> [--prompt p]   ← interactive CLI chat
 *
 * Environment:
 *   TERLLAMA_MODEL_DIR   model file directory
 *   TERLLAMA_PORT        server port (default 8375)
 */
#include "cli/commands.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

std::string home_dir() {
    const char* h = getenv("HOME");
    if (h) return h;
    struct passwd* pw = getpwuid(getuid());
    return pw ? pw->pw_dir : "/root";
}

std::string models_dir() {
    return home_dir() + "/.terllama/models";
}

std::string models_json_path() {
    return home_dir() + "/.terllama/models.json";
}

std::string fmt_size(double bytes) {
    char buf[32];
    if (bytes < 1024.0)
        snprintf(buf, sizeof(buf), "%.0f B", bytes);
    else if (bytes < 1024.0 * 1024.0)
        snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN — Subcommand dispatcher
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string cmd = argv[1];

    // ─── Subcommands ─────────────────────────────────────────────────────

    if (cmd == "list" || cmd == "ls") {
        return cmd_list();
    }

    if (cmd == "show") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " show <model>" << std::endl;
            return 1;
        }
        return cmd_show(argv[2]);
    }

    if (cmd == "rm" || cmd == "remove") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " rm <model>" << std::endl;
            return 1;
        }
        return cmd_rm(argv[2]);
    }

    if (cmd == "pull" || cmd == "download") {
        return cmd_pull(argc, argv);
    }

    if (cmd == "serve" || cmd == "server") {
        return cmd_serve(argc, argv);
    }

    if (cmd == "chat" || cmd == "cli") {
        return cmd_chat(argc, argv);
    }

    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage(argv[0]);
        return 0;
    }

    // ─── Legacy mode: terllama "prompt" [max_tokens] [temp] ──────────────
    {
        std::string prompt = argv[1];
        int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 40;
        float temperature = (argc > 3) ? std::stof(argv[3]) : 0.7f;
        return cmd_legacy(prompt, max_tokens, temperature);
    }
}

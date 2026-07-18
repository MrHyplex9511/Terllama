/*
 * commands.h — CLI subcommand declarations for Terllama
 *
 * All cmd_* functions handle a single subcommand.
 * Helpers home_dir/models_dir/models_json_path/fmt_size are defined
 * in main.cpp.
 */
#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <csignal>

#define TERLLAMA_VERSION "1.0.0"

// ═══════════════════════════════════════════════════════════════════════════
// SIGNAL HANDLING (defined in commands.cpp, installed in main.cpp)
// ═══════════════════════════════════════════════════════════════════════════
extern std::atomic<bool> g_interrupted;
extern "C" void handle_signal(int sig);

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS (defined in main.cpp)
// ═══════════════════════════════════════════════════════════════════════════
std::string home_dir();
std::string models_dir();
std::string models_json_path();
std::string fmt_size(double bytes);

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMANDS
// ═══════════════════════════════════════════════════════════════════════════
int cmd_list();
int cmd_show(const std::string& model_id);
int cmd_rm(const std::string& model_id);
int cmd_chat(int argc, char** argv);
int cmd_pull(int argc, char** argv);
int cmd_serve(int argc, char** argv);
int cmd_legacy(const std::string& prompt, int max_tokens, float temperature);
int cmd_bench();
void print_usage(const char* prog);

// ═══════════════════════════════════════════════════════════════════════════
// External entry points (defined in server.cpp / downloader.cpp)
// ═══════════════════════════════════════════════════════════════════════════
int server_main(int argc, char** argv);
int downloader_main(int argc, char** argv);

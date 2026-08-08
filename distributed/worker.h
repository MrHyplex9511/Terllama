/*
 * worker.h — Terllama distributed-inference worker.
 *
 * HTTP server (httplib, threaded) that owns one pipeline shard of an ALS
 * model and serves the /health, /load, /forward, /reset RPC endpoints from
 * protocol.h. All model state is guarded by a mutex (v1: single-stream,
 * one request at a time).
 */
#pragma once

#include <httplib.h>
#include <memory>
#include <mutex>
#include <string>

#include "protocol.h"
#include "shard_loader.h"

namespace tldist {

class Worker {
public:
    explicit Worker(std::string listen_addr);
    ~Worker() = default;

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // Loads a model shard from disk (throws on failure).
    void load_model(const std::string& model_path, const ShardSpec& spec);

    // Registers routes and serves until stop() is called (blocks).
    int run();

    // Stops the server (safe to call from a signal handler).
    void stop();

private:
    void register_routes();

    void handle_health(httplib::Response& res);
    void handle_load(const httplib::Request& req, httplib::Response& res);
    void handle_forward(const httplib::Request& req, httplib::Response& res);
    void handle_reset(httplib::Response& res);

    std::string listen_addr_;
    httplib::Server svr_;
    std::mutex mu_;
    std::shared_ptr<ShardModel> loaded_;  // nullptr until /load succeeds
    std::string model_dir_;
};

}  // namespace tldist

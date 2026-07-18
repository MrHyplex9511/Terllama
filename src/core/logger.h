/*
 * logger.h — Hierarchical threaded logger
 *
 * Levels: DEBUG < INFO < WARN < ERROR
 * Thread-safe via mutex. Outputs to stderr.
 */
#pragma once
#include <cstdio>
#include <mutex>
#include <string>

enum LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static LogLevel level;
    static std::mutex mutex;

    static void set_level(LogLevel lvl) { level = lvl; }

    template<typename... Args>
    static void debug(const char* fmt, Args... args) {
        if (DEBUG < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[DEBUG] ");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void info(const char* fmt, Args... args) {
        if (INFO < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[INFO]  ");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void warn(const char* fmt, Args... args) {
        if (WARN < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[WARN]  ");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void error(const char* fmt, Args... args) {
        if (ERROR < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[ERROR] ");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        fprintf(stderr, "\n");
    }
};

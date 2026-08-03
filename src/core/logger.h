/*
 * logger.h — Hierarchical threaded logger
 *
 * Levels: DEBUG < INFO < WARN < ERROR
 * Thread-safe via mutex. Outputs to stderr.
 *
 * Supports two format styles:
 *   - fmt-style:  Logger::info("value = {}", x);        (preferred)
 *   - printf:     Logger::info("value = %d", x);        (fallback)
 */
#pragma once
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>

enum LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static LogLevel level;
    static std::mutex mutex;

    static void set_level(LogLevel lvl) { level = lvl; }

    // ── fmt-style formatter ─────────────────────────────────────────────
    // Converts each arg to a string and substitutes {} placeholders in order.
    // If fmt contains no {}, falls back to printf-style fprintf.
private:
    static inline void append_arg(std::string&) {}

    template<typename T>
    static inline void append_arg(std::string& s, const T& v) {
        if constexpr (std::is_arithmetic_v<T> || std::is_convertible_v<T, std::string>) {
            std::ostringstream oss;
            oss << v;
            s += oss.str();
        } else {
            std::ostringstream oss;
            oss << v;
            s += oss.str();
        }
    }

    template<typename... Args>
    static inline std::string fmt_str(const char* fmt, Args... args) {
        std::string f = fmt ? fmt : "";
        // Only use fmt-style substitution if there's at least one {}
        if (f.find("{}") == std::string::npos) {
            return f; // signal: no placeholders → caller uses printf fallback
        }
        std::string out;
        size_t i = 0;
        auto consume = [&](auto& v) {
            size_t pos = f.find("{}", i);
            if (pos == std::string::npos) {
                // More args than placeholders: append them
                append_arg(out, v);
            } else {
                out.append(f, i, pos - i);
                append_arg(out, v);
                i = pos + 2;
            }
        };
        (consume(args), ...);
        out.append(f, i, std::string::npos);
        return out;
    }

    static inline bool has_placeholder(const char* fmt) {
        return fmt && std::string(fmt).find("{}") != std::string::npos;
    }

public:
    template<typename... Args>
    static void debug(const char* fmt, Args... args) {
        if (DEBUG < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[DEBUG] ");
        if (has_placeholder(fmt)) {
            fputs(fmt_str(fmt, args...).c_str(), stderr);
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
            fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        }
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void info(const char* fmt, Args... args) {
        if (INFO < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[INFO]  ");
        if (has_placeholder(fmt)) {
            fputs(fmt_str(fmt, args...).c_str(), stderr);
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
            fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        }
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void warn(const char* fmt, Args... args) {
        if (WARN < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[WARN]  ");
        if (has_placeholder(fmt)) {
            fputs(fmt_str(fmt, args...).c_str(), stderr);
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
            fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        }
        fprintf(stderr, "\n");
    }

    template<typename... Args>
    static void error(const char* fmt, Args... args) {
        if (ERROR < level) return;
        std::lock_guard<std::mutex> lock(mutex);
        fprintf(stderr, "[ERROR] ");
        if (has_placeholder(fmt)) {
            fputs(fmt_str(fmt, args...).c_str(), stderr);
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
            fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
        }
        fprintf(stderr, "\n");
    }
};

/*
 * logger.cpp — Logger static member definitions
 */
#include "logger.h"

LogLevel Logger::level = INFO;
std::mutex Logger::mutex;

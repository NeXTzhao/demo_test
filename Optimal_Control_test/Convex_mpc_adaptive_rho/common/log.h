#pragma once
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "binary.h"
#include "log_writer.h"

extern std::ostream dev_null;

#define LEFT_BRACKET  "["
#define RIGHT_BRACKET "]"

#ifndef MODULE_NAME
#define MODULE_NAME cyber::Binary::GetName().c_str()
#endif

#define ADEBUG ALOG_MODULE(MODULE_NAME, DEBUG)
#define AINFO  ALOG_MODULE(MODULE_NAME, INFO)
#define AWARN  ALOG_MODULE(MODULE_NAME, WARN)
#define AERROR ALOG_MODULE(MODULE_NAME, ERROR)
#define AFATAL ALOG_MODULE(MODULE_NAME, FATAL)

#ifndef ALOG_MODULE_STREAM
#define ALOG_MODULE_STREAM(log_severity) ALOG_MODULE_STREAM_##log_severity
#endif

#ifndef ALOG_MODULE
#define ALOG_MODULE(module, log_severity) \
    ALOG_MODULE_STREAM(log_severity)      \
    (module)
#endif

#define LOG_LEVEL_DEBUG 1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_WARN  3
#define LOG_LEVEL_ERROR 4
#define LOG_LEVEL_FATAL 5

#if PLANNING_DEBUG >= LOG_LEVEL_DEBUG
#define OUTPUT_PORT std::cout
#elif PLANNING_DEBUG >= LOG_LEVEL_ERROR
#define OUTPUT_PORT std::cerr
#else
#define OUTPUT_PORT dev_null
#endif

#define ALOG_MODULE_STREAM_DEBUG(module)                                        \
    if (PLANNING_DEBUG <= LOG_LEVEL_DEBUG)                                      \
    OUTPUT_PORT << PRINT_LIGHT_GRAY << "\n[DEBUG]" << __FILE__ << ':' << __LINE__ \
              << LEFT_BRACKET << module << RIGHT_BRACKET << PRINT_NONE

#define ALOG_MODULE_STREAM_INFO(module)                                        \
    if (PLANNING_DEBUG <= LOG_LEVEL_INFO)                                      \
    OUTPUT_PORT << PRINT_LIGHT_GRAY << "\n[INFO]" << __FILE__ << ':' << __LINE__ \
              << LEFT_BRACKET << module << RIGHT_BRACKET << PRINT_NONE

#define ALOG_MODULE_STREAM_WARN(module)                                   \
    if (PLANNING_DEBUG <= LOG_LEVEL_WARN)                                 \
    OUTPUT_PORT << PRINT_BROWN << "\n[WARN]" << __FILE__ << ':' << __LINE__ \
              << LEFT_BRACKET << module << RIGHT_BRACKET << PRINT_NONE

#define ALOG_MODULE_STREAM_ERROR(module)                                 \
    if (PLANNING_DEBUG <= LOG_LEVEL_ERROR)                               \
    OUTPUT_PORT << PRINT_RED << "\n[ERROR]" << __FILE__ << ':' << __LINE__ \
              << LEFT_BRACKET << module << RIGHT_BRACKET << PRINT_NONE

#define ALOG_MODULE_STREAM_FATAL(module)                                       \
    if (PLANNING_DEBUG <= LOG_LEVEL_FATAL)                                     \
    OUTPUT_PORT << PRINT_LIGHT_RED << "\n[FATAL]" << __FILE__ << ':' << __LINE__ \
              << LEFT_BRACKET << module << RIGHT_BRACKET << PRINT_NONE

#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)              \
    if (ptr == nullptr) {                \
        AWARN << #ptr << " is nullptr."; \
        return;                          \
    }
#endif

#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val)     \
    if (ptr == nullptr) {                \
        AWARN << #ptr << " is nullptr."; \
        return val;                      \
    }
#endif

#if !defined(RETURN_IF)
#define RETURN_IF(condition)               \
    if (condition) {                       \
        AWARN << #condition << " is met."; \
        return;                            \
    }
#endif

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)      \
    if (condition) {                       \
        AWARN << #condition << " is met."; \
        return val;                        \
    }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
    if (ptr == nullptr) {             \
        return (val);                 \
    }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
    if (condition) {                   \
        return (val);                  \
    }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
    if (condition) {          \
        return;               \
    }
#endif

#define PRINT_NONE         "\033[m"
#define PRINT_RED          "\033[0;32;31m"
#define PRINT_LIGHT_RED    "\033[1;31m"
#define PRINT_GREEN        "\033[0;32;32m"
#define PRINT_LIGHT_GREEN  "\033[1;32m"
#define PRINT_BLUE         "\033[0;32;34m"
#define PRINT_LIGHT_BLUE   "\033[1;34m"
#define PRINT_DARY_GRAY    "\033[1;30m"
#define PRINT_CYAN         "\033[0;36m"
#define PRINT_LIGHT_CYAN   "\033[1;36m"
#define PRINT_PURPLE       "\033[0;35m"
#define PRINT_LIGHT_PURPLE "\033[1;35m"
#define PRINT_BROWN        "\033[0;33m"
#define PRINT_YELLOW       "\033[1;33m"
#define PRINT_LIGHT_GRAY   "\033[0;37m"
#define PRINT_WHITE        "\033[1;37m"

#define ACHECK(cond)          CHECK(cond)
#define DCHECK(condition)     CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#define DCHECK_NOTNULL(val)   CHECK_NOTNULL(val)

#define CHECK(cond)          Check(cond, #cond, __FILE__, __LINE__)
#define CHECK_EQ(val1, val2) CheckEq(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_NE(val1, val2) CheckNe(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_LE(val1, val2) CheckLe(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_LT(val1, val2) CheckLt(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_GE(val1, val2) CheckGe(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_GT(val1, val2) CheckGt(val1, val2, #val1, #val2, __FILE__, __LINE__)
#define CHECK_NOTNULL(val)   CheckNotNull(val, #val, __FILE__, __LINE__)

template <typename T>
inline bool Check(T condition, const char* name, const char* file, int line) {
    if (!(condition)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s \n" PRINT_NONE, file, line, name);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckEq(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 == val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s == %s\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckNe(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 != val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s != %s.\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckLe(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 <= val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s <= %s.\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckLt(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 < val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s < %s.\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckGe(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 >= val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s >= %s.\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T1, typename T2>
inline bool CheckGt(T1 val1, T2 val2, const char* name1, const char* name2, const char* file, int line) {
    if (!(val1 > val2)) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s > %s.\n" PRINT_NONE, file, line, name1, name2);
        return false;
    }
    return true;
}

template <typename T>
inline T CheckNotNull(T&& t, const char* name, const char* file, int line) {
    if (!t) {
        fprintf(stderr, PRINT_RED "\n%s:%d Check failed: %s Must be non NULL.\n" PRINT_NONE, file, line, name);
    }
    return std::forward<T>(t);
}

/**
 * @brief a simple wrapper for spdlog that supports close log
 *
 */
class xlogger {
private:
#ifdef PLANNING_DEBUG
    std::shared_ptr<spdlog::logger> spdlogger_;
#endif

public:
    xlogger(const std::string& logger_name, const std::string& filename,
            bool truncate = false)
#ifdef PLANNING_DEBUG
        : spdlogger_(spdlog::basic_logger_mt(logger_name, filename, truncate))
#endif
    {
    }

    template <typename... Args>
//    inline void info(fmt::format_string<Args...> fmt, Args&&... args) {
//#ifdef PLANNING_DEBUG
//        spdlogger_->info(fmt, std::forward<Args>(args)...);
//#endif
//    }

    inline void flush() {
#ifdef PLANNING_DEBUG
        spdlogger_->flush();
#endif
    }

    inline void drop_all() {
#ifdef PLANNING_DEBUG
        spdlog::drop_all();
#endif
    }
};
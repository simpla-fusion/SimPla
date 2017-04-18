/**
 *  @file  log.cpp
 *
 * @date    2014-7-29  AM8:43:27
 * @author salmon
 */

#include "Log.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "SingletonHolder.h"

namespace simpla {
namespace logger {

/**
 *  @ingroup Logging
 *  \brief Logging stream, should be used  as a singleton
 */
struct LoggerStreams {
    static constexpr unsigned int DEFAULT_LINE_WIDTH = 120;
    bool is_opened_ = false;
    int line_width_ = DEFAULT_LINE_WIDTH;
    int mpi_rank_ = 0, mpi_size_ = 1;

    LoggerStreams(int level = LOG_INFORM) : m_std_out_level_(level), line_width_(DEFAULT_LINE_WIDTH) {}
    ~LoggerStreams() { close(); }

    void init();

    void close();

    inline void open_file(std::string const &name) {
        if (fs.is_open()) fs.close();
        fs.open(name.c_str(), std::ios_base::trunc);
    }

    void push(int level, std::string const &msg);

    inline void set_stdout_level(int l) { m_std_out_level_ = l; }

    int get_line_width() const { return line_width_; }

    void set_line_width(int lineWidth) { line_width_ = lineWidth; }

    static std::string time_stamp() {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        char mtstr[100];
        std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

        return std::string(mtstr);
    }

   private:
    int m_std_out_level_;
    std::ofstream fs;
};

void LoggerStreams::init() { is_opened_ = true; }

void LoggerStreams::close() {
    if (is_opened_) {
        VERBOSE << "LoggerStream is closed!" << std::endl;
        if (m_std_out_level_ >= LOG_INFORM && mpi_rank_ == 0) { std::cout << std::endl; }
        if (fs.is_open()) { fs.close(); }
        is_opened_ = false;
    }
}

void LoggerStreams::push(int level, std::string const &msg) {
    if (msg == "" || ((level == LOG_MESSAGE) && mpi_rank_ > 0)) return;

    std::ostringstream prefix;

    std::string surfix("");

    switch (level) {
        case LOG_FORCE_OUTPUT:
        case LOG_OUT_RANGE_ERROR:
        case LOG_LOGIC_ERROR:
        case LOG_ERROR:
            prefix << "[E]";
            break;
        case LOG_WARNING:
            prefix << "[W]";  // red
            break;
        case LOG_LOG:
            prefix << "[L]";
            break;
        case LOG_VERBOSE:
            prefix << "[V]";
            break;
        case LOG_INFORM:
            prefix << "[I]";
            break;
        case LOG_DEBUG:
            prefix << "[D]";
            break;
        default:
            break;
    }

    prefix << "[" << time_stamp() << "] ";

    //    if (mpi_size_ > 1) {
    prefix << "[" << mpi_rank_ << "/" << mpi_size_ << "]";
    //    }

    if (level <= m_std_out_level_) {
        switch (level) {
            case LOG_FORCE_OUTPUT:
            case LOG_OUT_RANGE_ERROR:
            case LOG_LOGIC_ERROR:
            case LOG_ERROR:
                std::cerr << "\e[1;31m" << std::setw(35) << prefix.str() << "\e[1;37m" << msg << "\e[0m" << surfix;
                break;
            case LOG_WARNING:
                std::cerr << "\e[1;32m" << std::setw(35) << prefix.str() << "\e[1;37m" << msg << "\e[0m" << surfix;
                break;
            case LOG_MESSAGE:
                std::cout << msg;
                break;
            default:
                std::cout << std::setw(35) << std::left << prefix.str() << msg << surfix;
        }
    }

    if (!fs.good()) open_file("simpla.log");

    fs << std::endl << prefix.str() << msg << surfix;
}

void open_file(std::string const &file_name) { return SingletonHolder<LoggerStreams>::instance().open_file(file_name); }

void close() { SingletonHolder<LoggerStreams>::instance().close(); }

void set_stdout_level(int l) { return SingletonHolder<LoggerStreams>::instance().set_stdout_level(l); }

void set_mpi_comm(int r, int s) {
    SingletonHolder<LoggerStreams>::instance().mpi_rank_ = r;
    SingletonHolder<LoggerStreams>::instance().mpi_size_ = s;
}

void set_line_width(int lw) { return SingletonHolder<LoggerStreams>::instance().set_line_width(lw); }

int get_line_width() { return SingletonHolder<LoggerStreams>::instance().get_line_width(); }

Logger::Logger() : base_type(), m_level_(0), current_line_char_count_(0), endl_(true) {}

Logger::Logger(int lv) : m_level_(lv), current_line_char_count_(0), endl_(true) {
    base_type::operator<<(std::boolalpha);

    current_line_char_count_ = get_buffer_length();
}

Logger::~Logger() {
    switch (m_level_) {
        case LOG_ERROR_RUNTIME:
            throw(std::runtime_error(this->str()));
            break;
        case LOG_ERROR_BAD_CAST:
            flush();
            throw(std::bad_cast());
            break;
        case LOG_ERROR_OUT_OF_RANGE:
            throw(std::out_of_range(this->str()));
            break;
        case LOG_ERROR_LOGICAL:
            throw(std::logic_error(this->str()));
            break;
        default:
            flush();
    }
}

int Logger::get_buffer_length() const { return static_cast<int>(this->str().size()); }

void Logger::flush() {
    SingletonHolder<LoggerStreams>::instance().push(m_level_, this->str());
    this->str("");
}

void Logger::surffix(std::string const &s) {
    (*this) << std::setfill('.')
            << std::setw(SingletonHolder<LoggerStreams>::instance().get_line_width() - current_line_char_count_)
            << std::right << s << std::left << std::endl;

    flush();
}

void Logger::endl() {
    (*this) << std::endl;
    current_line_char_count_ = 0;
    endl_ = true;
}

void Logger::not_endl() { endl_ = false; }

}  // namespace logger
}
// namespace simpla

/**
 *  @file  log.cpp
 *
 * @date    2014-7-29  AM8:43:27
 * @author salmon
 */

#include "log.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../design_pattern/singleton_holder.h"
#include "parse_command_line.h"

namespace simpla {
namespace gtl {
namespace logger {

/**
 *  @ingroup Logging
 *  \brief Logging stream, should be used  as a singleton
 */
struct LoggerStreams //: public SingletonHolder<LoggerStreams>
{
    bool is_opened_ = false;
    size_t line_width_;

    int mpi_rank_ = 0, mpi_size_ = 1;

    static constexpr unsigned int DEFAULT_LINE_WIDTH = 100;

    LoggerStreams(int level = LOG_INFORM)
            : std_out_visable_level_(level), line_width_(DEFAULT_LINE_WIDTH)
    {
    }

    ~LoggerStreams()
    {
        close();
    }

    std::string init(int argc, char **argv);

    void close();

    inline void open_file(std::string const &name)
    {
        if (fs.is_open())
            fs.close();

        fs.open(name.c_str(), std::ios_base::trunc);
    }

    void put(int level, std::string const &msg);

    inline void set_stdout_visable_level(int l)
    {
        std_out_visable_level_ = l;
    }

    size_t get_line_width() const
    {
        return line_width_;
    }

    void set_line_width(size_t lineWidth)
    {
        line_width_ = lineWidth;
    }

    static std::string time_stamp()
    {

        auto now = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now());

        char mtstr[100];
        std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

        return std::string(mtstr);
    }

private:
    int std_out_visable_level_;

    std::ofstream fs;

};

std::string LoggerStreams::init(int argc, char **argv)
{

    parse_cmd_line(argc, argv,

                   [&, this](std::string const &opt, std::string const &value) -> int
                       {
                       if (opt == "log")
                       {
                           this->open_file(value);
                       }
                       else if (opt == "V" || opt == "verbose")
                       {
                           this->set_stdout_visable_level(std::atoi(value.c_str()));
                       }
                       else if (opt == "quiet")
                       {
                           this->set_stdout_visable_level(LOG_INFORM - 1);
                       }
                       else if (opt == "log_width")
                       {
                           this->set_line_width(std::atoi(value.c_str()));
                       }

                       return CONTINUE;
                       }

    );

    is_opened_ = true;

    VERBOSE << "LoggerStream is initialized!" << std::endl;

    return "\t-V,\t--verbose <NUM> \t, Verbose mode.  Print debugging messages, \n"
            "\t\t\t\t\t   <-10 means quiet, >10 means print as much as it can. default=0 \n";

}

void LoggerStreams::close()
{

    if (is_opened_)
    {
        VERBOSE << "LoggerStream is closed!" << std::endl;
        if (std_out_visable_level_ >= LOG_INFORM && mpi_rank_ == 0)
            std::cout << std::endl;

        if (fs.is_open())
            fs.close();

        is_opened_ = false;
    }

}

void LoggerStreams::put(int level, std::string const &msg)
{

    if (msg == ""
        || ((level == LOG_INFORM || level == LOG_MESSAGE) && mpi_rank_ > 0))
        return;

    std::ostringstream prefix;

    std::string surfix("");

    switch (level)
    {
        case LOG_FORCE_OUTPUT:
        case LOG_OUT_RANGE_ERROR:
        case LOG_LOGIC_ERROR:
        case LOG_ERROR:
            prefix << "[E]";
            break;
        case LOG_WARNING:
            prefix << "[W]"; //red
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
    }
    if (mpi_size_ > 1)
    {
        prefix << "[" << mpi_rank_ << "/" << mpi_size_ << "]";
    }

    prefix << "[" << time_stamp() << "]";

    if (level <= std_out_visable_level_)
    {
        switch (level)
        {
            case LOG_FORCE_OUTPUT:
            case LOG_OUT_RANGE_ERROR:
            case LOG_LOGIC_ERROR:
            case LOG_ERROR:
                std::cerr << "\e[1;31m" << prefix.str() << "\e[1;37m" << msg
                << "\e[0m" << surfix;
                break;
            case LOG_WARNING:
                std::cerr << "\e[1;32m" << prefix.str() << "\e[1;37m" << msg
                << "\e[0m" << surfix;
                break;
            case LOG_MESSAGE:
                std::cout << msg;
                break;
            default:
                std::cout << prefix.str() << msg << surfix;
        }

    }

    if (!fs.good())
        open_file("simpla.log");

    fs << std::endl << prefix.str() << msg << surfix;

}

Logger::Logger()
        : level_(0), current_line_char_count_(0), endl_(true)
{
}

Logger::Logger(Logger const
& r)
:
level_(r
.level_),
current_line_char_count_(
        r
.current_line_char_count_),
endl_(r
.endl_) {
}

Logger::Logger(Logger && r)
        : level_(r.level_), current_line_char_count_(
        r.current_line_char_count_), endl_(r.endl_)
{
}

Logger::Logger(int lv)
        : level_(lv), current_line_char_count_(0), endl_(true)
{
    buffer_ << std::boolalpha;

    current_line_char_count_ = get_buffer_length();
}

Logger::~Logger()
{
    SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());
}

std::string Logger::init(int argc, char **argv)
{
    return SingletonHolder<LoggerStreams>::instance().init(argc, argv);
}

void Logger::set_stdout_visable_level(int l)
{
    return SingletonHolder<LoggerStreams>::instance().set_stdout_visable_level(
            l);
}

void Logger::set_mpi_comm(int r, int s)
{
    SingletonHolder<LoggerStreams>::instance().mpi_rank_ = r;
    SingletonHolder<LoggerStreams>::instance().mpi_size_ = s;
}

size_t Logger::get_buffer_length() const
{
    return buffer_.str().size();
}

size_t Logger::get_line_width() const
{
    return SingletonHolder<LoggerStreams>::instance().get_line_width();
}

void Logger::flush()
{
    SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());
    buffer_.str("");
}

void Logger::surffix(std::string const &s)
{
    const_cast<this_type *>(this)->buffer_ << std::setfill('.')

    << std::setw(
            SingletonHolder<LoggerStreams>::instance().get_line_width()
            - current_line_char_count_)

    << std::right << s << std::left << std::endl;

    flush();
}

void Logger::endl()
{
    this->buffer_ << std::endl;
    current_line_char_count_ = 0;
    endl_ = true;
}

void Logger::not_endl()
{
    endl_ = false;
}

std::string init_logger(int argc, char **argv)
{
    return SingletonHolder<LoggerStreams>::instance().init(argc, argv);
}

void close_logger()
{
    SingletonHolder<LoggerStreams>::instance().close();
}

}  // namespace logger
}
}//  namespace simpla::gtl


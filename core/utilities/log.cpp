/**
 * \file log.cpp
 *
 * \date    2014年7月29日  上午8:43:27 
 * \author salmon
 */

#include "log.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "misc_utilities.h"
#include "singleton_holder.h"
#include "parse_command_line.h"
namespace simpla
{
/**
 *  \ingroup Logging
 *  \brief Logging stream, shuold be used  as a singleton
 */
struct LoggerStreams //: public SingletonHolder<LoggerStreams>
{
	bool is_opened_ = false;
	size_t line_width_;

	size_t mpi_rank_ = 0, mpi_size_ = 1;

	static constexpr unsigned int DEFAULT_LINE_WIDTH = 100;

	LoggerStreams(int level = LOG_INFORM) :
			std_out_visable_level_(level), line_width_(DEFAULT_LINE_WIDTH)
	{
	}
	~LoggerStreams()
	{

		close();

	}

	void init(int argc, char** argv);
	void close();

	inline void open_file(std::string const & name)
	{
		if (fs.is_open())
			fs.close();

		fs.open(name.c_str(), std::ios_base::trunc);
	}

	void put(int level, std::string const & msg);

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

void LoggerStreams::init(int argc, char** argv)
{

	bool show_help = false;

	parse_cmd_line(argc, argv,

	[&,this](std::string const & opt,std::string const & value)->int
	{
		if( opt=="log")
		{
			this->open_file (value);
		}
		else if(opt=="v" || opt=="verbose")
		{
			this->set_stdout_visable_level(ToValue<int>(value));
		}
		else if( opt=="quiet")
		{
			this->set_stdout_visable_level(LOG_INFORM-1);
		}
		else if( opt=="log_width")
		{
			this->set_line_width(ToValue<int>(value));
		}
		else if(opt=="h"|| opt=="help")
		{
			show_help = true;
		}
		return CONTINUE;
	}

	);

	if (show_help)
	{
		SHOW_OPTIONS("-v,--verbose <NUM> ", "Verbose mode")
	}
	is_opened_ = true;
	VERBOSE << "LoggerStream is initialized!" << std::endl;

}
void LoggerStreams::close()
{

	if (is_opened_)
	{
		VERBOSE << "LoggerStream is closed!" << std::endl;
		if (std_out_visable_level_ >= LOG_INFORM)
			std::cout << std::endl;

		if (fs.is_open())
			fs.close();

		is_opened_ = false;
	}

}
void LoggerStreams::put(int level, std::string const & msg)
{

	if (msg == "" || (level == LOG_INFORM && mpi_rank_ > 0))
		return;

	std::string prefix(""), surfix("");

	switch (level)
	{
	case LOG_FORCE_OUTPUT:
	case LOG_OUT_RANGE_ERROR:
	case LOG_LOGIC_ERROR:
	case LOG_ERROR:
		prefix = "[E]";
		break;
	case LOG_WARNING:
		prefix = "[W]"; //red
		break;
	case LOG_LOG:
		prefix = "[L]";
		break;
	case LOG_VERBOSE:
		prefix = "[V]";
		break;
	case LOG_INFORM:
		prefix = "[I]";
		break;
	case LOG_DEBUG:
		prefix = "[D]";
		break;
	}
	if (mpi_size_ > 1)
	{
		prefix += "[" + ToString(mpi_rank_) + "/" + ToString(mpi_size_) + "]";
	}

	prefix += "[" + time_stamp() + "]";

	if (!fs.good())
		open_file("simpla.log");

	// @bug  can not write SimPla log to file

	fs << std::endl << prefix << msg << surfix;
	;

	if (level <= std_out_visable_level_)
	{
		switch (level)
		{
		case LOG_FORCE_OUTPUT:
		case LOG_OUT_RANGE_ERROR:
		case LOG_LOGIC_ERROR:
		case LOG_ERROR:
			std::cerr << "\e[1;31m" << prefix << "\e[1;37m" << msg << "\e[0m"
					<< surfix;
			break;
		case LOG_WARNING:
			std::cerr << "\e[1;32m" << prefix << "\e[1;37m" << msg << "\e[0m"
					<< surfix;
			break;
		case LOG_STDOUT:
			std::cout << msg;
			break;
		default:
			std::cout << prefix << msg << surfix;
		}

	}

}

Logger::Logger() :
		level_(0), current_line_char_count_(0), endl_(true)
{
}

Logger::Logger(Logger const & r) :
		level_(r.level_), current_line_char_count_(r.current_line_char_count_), endl_(
				r.endl_)
{
}

Logger::Logger(Logger && r) :
		level_(r.level_), current_line_char_count_(r.current_line_char_count_), endl_(
				r.endl_)
{
}
Logger::Logger(int lv) :
		level_(lv), current_line_char_count_(0), endl_(true)
{
	buffer_ << std::boolalpha;

	current_line_char_count_ = get_buffer_length();
}

Logger::~Logger()
{
	SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());
}

void Logger::init(int argc, char** argv)
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

void Logger::surffix(std::string const & s)
{
	const_cast<this_type*>(this)->buffer_ << std::setfill('.')

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

void init_logger(int argc, char**argv)
{
	SingletonHolder<LoggerStreams>::instance().init(argc, argv);
}
void close_logger()
{
	SingletonHolder<LoggerStreams>::instance().close();
}
}
// namespace simpla


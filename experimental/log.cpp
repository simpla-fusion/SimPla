/**
 * \file log.cpp
 *
 * \date    2014年7月17日  上午8:27:49 
 * \author salmon
 */

#include "log.h"

#include "singleton_holder.h"
#include <ios>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <bitset>
#include <cassert>
#ifdef USE_MPI
#	include "../parallel/message_comm.h"
#endif
#include "parse_command_line.h"
#include "utilities.h"
#include "properties.h"
namespace simpla
{
/**
 *  \ingroup Logging
 *  \brief Logging stream, shuold be used  as a singleton
 */
class LoggerStreams //: public SingletonHolder<LoggerStreams>
{
	size_t indent_;
	Properties properties;

public:

	LoggerStreams(int level = LOG_INFORM)
			: indent_(0)
	{
		properties["Ling Width"] = 100; // DEFAULT_LINE_WIDTH;
		properties["Visable Level"] = 0;

	}
	~LoggerStreams()
	{
		fs.close();
	}

	void init(int argc, char** argv)
	{

		ParseCmdLine(argc, argv,

		[&,this](std::string const & opt,std::string const & value)->int
		{
			if( opt=="log")
			{
				this->open_file (value);
			}
			else if(opt=="v")
			{
				properties["Visable Level"]=(ToValue<int>(value));
			}
			else if( opt=="verbose")
			{
				properties["Visable Level"]=int(LOG_VERBOSE);
			}
			else if( opt=="quiet")
			{
				properties["Visable Level"]=int(LOG_INFORM-1);
			}
			else if( opt=="log_width")
			{
				properties["Ling Width"]=(ToValue<int>(value));
			}
			return CONTINUE;
		}

		);

	}
	void set_property(std::string const & name, Any const&v)
	{
		properties[name] = v;
	}
	Any const & get_property_any(std::string const &name) const
	{
		return properties[name].template as<Any>();
	}
	inline void open_file(std::string const & name)
	{
		if (fs.is_open())
			fs.close();

		fs.open(name.c_str(), std::ios_base::out);
	}

	void IncreaseIndent(size_t n = 1)
	{
		indent_ += n;
	}
	void DecreaseIndent(size_t n = 1)
	{
		if (indent_ > n)
			indent_ -= n;
		else
			indent_ = 0;
	}
	size_t get_indent() const
	{
		return indent_;
	}

	static std::string time_stamp()
	{

		auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

		char mtstr[100];
		std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

		return std::string(mtstr);
	}

	void put(int level, std::string const & msg)
	{

		if (msg == "" || (level == LOG_INFORM && GLOBAL_COMM.get_rank()>0) ) return;

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
		};

		if(level != LOG_INFORM)
		{
			prefix+="[" + ToString(GLOBAL_COMM.get_rank()) + "/" + ToString(GLOBAL_COMM.get_size())+ "]";
		}
		prefix+="[" + time_stamp() + "]";

		if (!fs.good()) open_file("simpla.log");

		if (fs.good()) fs << prefix << msg << surfix;;

		if (level <= properties["Visable Level"].template as< int >(0))
		{
			switch (level)
			{
				case LOG_FORCE_OUTPUT:
				case LOG_OUT_RANGE_ERROR:
				case LOG_LOGIC_ERROR:
				case LOG_ERROR:
				std::cerr <<"\e[1;31m"<< prefix <<"\e[1;37m"<< msg <<"\e[0m"<< surfix;
				break;
				case LOG_WARNING:
				std::cerr <<"\e[1;32m"<< prefix <<"\e[1;37m"<< msg <<"\e[0m"<< surfix;
				break;
				case LOG_INFORM:
				std::cout <<"\e[1;32m"<< prefix <<"\e[1;37m"<< msg <<"\e[0m"<< surfix;
				break;
				default:
				std::cout << prefix << msg << surfix;
			}

		}

	}
private:

	std::fstream fs;

};

Logger::Logger()
		: null_dump_(true), level_(0), current_line_char_count_(0), indent_(0), endl_(true)
{
}

Logger::Logger(Logger const & r)
		: null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
		        r.indent_), endl_(r.endl_)
{
}

Logger::Logger(Logger && r)
		: null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
		        r.indent_), endl_(r.endl_)
{
}

Logger::Logger(int lv, size_t indent)
		: null_dump_(false), level_(lv), current_line_char_count_(0), indent_(indent), endl_(true)
{
	buffer_ << std::boolalpha;

	size_t indent_width = SingletonHolder<LoggerStreams>::instance().get_indent();
	if (indent_width > 0)
		buffer_ << std::setfill('-') << std::setw(indent_width) << "+";

	set_indent(indent_);

	current_line_char_count_ = get_buffer_length();
}

Logger::~Logger()
{
	if (null_dump_)
		return;

	if (current_line_char_count_ > 0 && endl_)
		buffer_ << std::endl;
	SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());

	unset_indent(indent_);
}

void Logger::init(int argc, char** argv)
{
	SingletonHolder<LoggerStreams>::instance().init(argc, argv);
}

void Logger::set_property_(std::string const & name, Any const &v)
{
	SingletonHolder<LoggerStreams>::instance().set_property(name, v);
}
Any Logger::get_property_(std::string const & name) const
{
	return SingletonHolder<LoggerStreams>::instance().get_property_any(name);
}
void Logger::set_indent(size_t n)
{
	SingletonHolder<LoggerStreams>::instance().IncreaseIndent(n);
	indent_ += n;
}
void Logger::unset_indent(size_t n)
{
	if (indent_ >= n)
		SingletonHolder<LoggerStreams>::instance().DecreaseIndent(n);
	indent_ -= n;
}
void Logger::time_stamp()
{
	*this << SingletonHolder<LoggerStreams>::instance().time_stamp();
}
size_t Logger::get_buffer_length() const
{
	return buffer_.str().size();
}
void Logger::flush()
{
	SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());
	buffer_.str("");
}

void Logger::surffix(std::string const & s)
{
	const_cast<this_type*>(this)->buffer_ << std::setfill('.')

	<< std::setw(get_property<int>("Line Width") - current_line_char_count_)

	<< std::right << s << std::left;

	endl();

	flush();
}

void Logger::endl()
{
	const_cast<this_type*>(this)->buffer_ << std::endl;
	current_line_char_count_ = 0;
	endl_ = true;
}
void Logger::not_endl()
{
	endl_ = false;
}

Logger & Logger::operator<<(LoggerStreamManipulator manip)
{
	// call the function, and return it's value
	return manip(*this);
}

// take in a function with the custom signature
Logger const& Logger::operator<<(LoggerStreamConstManipulator manip) const
{
	// call the function, and return it's value
	return manip(*this);
}

//! define an operator<< to take in std::endl
Logger const& Logger::operator<<(StandardEndLine manip) const
{
	// call the function, but we cannot return it's value
	manip(const_cast<this_type*>(this)->buffer_);
	return *this;
}

Logger & Logger::operator<<(StandardEndLine manip)
{
	// call the function, but we cannot return it's value
	manip(const_cast<this_type*>(this)->buffer_);
	return *this;
}

}
// namespace simpla

/*  ____  _           ____  _
 * / ___|(_)_ __ ___ |  _ \| | __ _
 * \___ \| | '_ ` _ \| |_) | |/ _` |
 *  ___) | | | | | | |  __/| | (_| |
 * |____/|_|_| |_| |_|_|   |_|\__,_|
 *
 *
 *
 *
 * log.h
 *
 *  Created on: 2012-3-21
 *      Author: salmon
 */

#ifndef LOG_H_
#define LOG_H_

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
#ifdef USE_MPI
#	include "../parallel/message_comm.h"
#endif
#include "../utilities/parse_command_line.h"
#include "../utilities/utilities.h"

namespace simpla
{

enum
{
	LOG_FORCE_OUTPUT = -10000,

	LOG_OUT_RANGE_ERROR = -4, LOG_LOGIC_ERROR = -3, LOG_ERROR = -2,

	LOG_WARNING = -1,

	LOG_INFORM = 0, LOG_LOG = 1, LOG_VERBOSE = 11, LOG_DEBUG = -20
};
class LoggerStreams //: public SingletonHolder<LoggerStreams>
{
	size_t line_width_;
	size_t indent_;

public:
	static constexpr int DEFAULT_LINE_WIDTH = 100;

	LoggerStreams(int l = 0)
			: std_out_visable_level_(l), line_width_(DEFAULT_LINE_WIDTH), indent_(0)
	{
	}
	~LoggerStreams()
	{
		fs.close();
	}

	void Init(int argc, char** argv)
	{

		ParseCmdLine(argc, argv,

		[&,this](std::string const & opt,std::string const & value)->int
		{
			if( opt=="log")
			{
				this->OpenFile (value);
			}
			else if(opt=="v")
			{
				this->SetStdOutVisableLevel(ToValue<int>(value));
			}
			else if( opt=="verbose")
			{
				this->SetStdOutVisableLevel(LOG_VERBOSE);
			}
			else if( opt=="quiet")
			{
				this->SetStdOutVisableLevel(LOG_INFORM-1);
			}
			else if( opt=="log_width")
			{
				this->SetLineWidth(ToValue<int>(value));
			}
			return CONTINUE;
		}

		);

	}

	inline void OpenFile(std::string const & name)
	{
		if (fs.is_open())
			fs.close();

		fs.open(name.c_str(), std::ios_base::out);
	}

	void put(int level, std::string const & msg)
	{
		if (msg != "" && (!(level == LOG_INFORM
#ifdef USE_MPI
				&& GLOBAL_COMM.GetRank()>0
#endif
		)))
		{
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

			if (fs.good())
			{
				fs << msg;
			}
			else
			{
				OpenFile("simpla.log");
			}

			if (level <= std_out_visable_level_)
			{
				surfix += "\e[0m";

				switch (level)
				{
					case LOG_FORCE_OUTPUT:
					case LOG_OUT_RANGE_ERROR:
					case LOG_LOGIC_ERROR:
					case LOG_ERROR:
					prefix = "\e[1;31m" + prefix + "\e[1;37m"; //red
					break;
					case LOG_WARNING:
					prefix = "\e[1;32m" + prefix + "\e[1;37m";//red
					break;

				}

				std::cerr << prefix << msg << surfix;

			}

		}

	}
	inline void SetStdOutVisableLevel(int l)
	{
		std_out_visable_level_ = l;
	}

	size_t GetLineWidth() const
	{
		return line_width_;
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
	size_t GetIndent() const
	{
		return indent_;
	}

	void SetLineWidth(size_t lineWidth)
	{
		line_width_ = lineWidth;
	}

private:
	int std_out_visable_level_;

	std::fstream fs;

};

/***
 *
 *  log message buffer,
 *
 */
class Logger
{
	bool null_dump_;
	typedef std::ostringstream buffer_type;
	int level_;
	std::ostringstream buffer_;
	size_t current_line_char_count_;
	size_t indent_;
	bool endl_;
public:
	typedef Logger this_type;

	Logger()
			: null_dump_(true), level_(0), current_line_char_count_(0), indent_(0), endl_(true)
	{
	}

	Logger(Logger const & r)
			: null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
			        r.indent_), endl_(r.endl_)
	{
	}

	Logger(Logger && r)
			: null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
			        r.indent_), endl_(r.endl_)
	{
	}

	Logger(int lv, size_t indent = 0)
			: null_dump_(false), level_(lv), current_line_char_count_(0), indent_(indent), endl_(true)
	{
		buffer_ << std::boolalpha;

#ifdef USE_MPI
		buffer_ << "[" << GLOBAL_COMM.GetRank() << "/" << GLOBAL_COMM.GetSize()
		<< "]";
#endif
//		if (level_ == LOG_INFORM || level_ == LOG_LOG || level_ == LOG_VERBOSE)

		buffer_ << "[" << TimeStamp() << "]" << " ";

		size_t indent_width = SingletonHolder<LoggerStreams>::instance().GetIndent();
		if (indent_width > 0)
			buffer_ << std::setfill('-') << std::setw(indent_width) << "+";

		SetIndent(indent_);

		current_line_char_count_ = GetBufferLength();
	}

	~Logger()
	{
		if (null_dump_)
			return;

		if (current_line_char_count_ > 0 && endl_)
			buffer_ << std::endl;
		SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());

		UnsetIndent(indent_);
	}

	void SetIndent(size_t n = 1)
	{
		SingletonHolder<LoggerStreams>::instance().IncreaseIndent(n);
		indent_ += n;
	}
	void UnsetIndent(size_t n = 1)
	{
		if (indent_ >= n)
			SingletonHolder<LoggerStreams>::instance().DecreaseIndent(n);
		indent_ -= n;
	}

	size_t GetBufferLength() const
	{
		return buffer_.str().size();
	}
	void flush()
	{
		SingletonHolder<LoggerStreams>::instance().put(level_, buffer_.str());
		buffer_.str("");
	}

	void surffix(std::string const & s)
	{
		const_cast<this_type*>(this)->buffer_ << std::setfill('.')

		<< std::setw(SingletonHolder<LoggerStreams>::instance().GetLineWidth() - current_line_char_count_)

		<< std::right << s << std::left;

		endl();

		flush();
	}

	void endl()
	{
		const_cast<this_type*>(this)->buffer_ << std::endl;
		current_line_char_count_ = 0;
		endl_ = true;
	}
	void not_endl()
	{
		endl_ = false;
	}

	template<typename T> inline this_type & operator<<(T const& value)
	{
		if (null_dump_)
			return *this;

		current_line_char_count_ -= GetBufferLength();

		const_cast<this_type*>(this)->buffer_ << value;

		current_line_char_count_ += GetBufferLength();

		if (current_line_char_count_ > SingletonHolder<LoggerStreams>::instance().GetLineWidth())
			endl();

		return *this;
	}

	typedef Logger & (*LoggerStreamManipulator)(Logger &);

	Logger & operator<<(LoggerStreamManipulator manip)
	{
		// call the function, and return it's value
		return manip(*this);
	}

	typedef Logger & (*LoggerStreamConstManipulator)(Logger const &);

	// take in a function with the custom signature
	Logger const& operator<<(LoggerStreamConstManipulator manip) const
	{
		// call the function, and return it's value
		return manip(*this);
	}

//	// define the custom endl for this stream.
//	// note how it matches the `LoggerStreamManipulator`
//	// function signature
//	static this_type& endl(this_type& stream)
//	{
//		// print a new line
//		std::cout << std::endl;
//
//		// do other stuff with the stream
//		// std::cout, for example, will flush the stream
//		stream << "Called Logger::endl!" << std::endl;
//
//		return stream;
//	}

// this is the function signature of std::endl
	typedef std::basic_ostream<char, std::char_traits<char> > StdCoutType;
	typedef StdCoutType& (*StandardEndLine)(StdCoutType&);

// define an operator<< to take in std::endl
	this_type const& operator<<(StandardEndLine manip) const
	{
		// call the function, but we cannot return it's value
		manip(const_cast<this_type*>(this)->buffer_);
		return *this;
	}

	this_type & operator<<(StandardEndLine manip)
	{
		// call the function, but we cannot return it's value
		manip(const_cast<this_type*>(this)->buffer_);
		return *this;
	}

	static std::string TimeStamp()
	{

		auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

		char mtstr[100];
		std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

		return std::string(mtstr);
	}
private:
};

#define LOG_STREAM SingletonHolder<LoggerStreams>::instance()

#define WARNING Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define INFORM Logger(LOG_INFORM)

#define UNIMPLEMENT Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "Sorry, this function is not implemented. Try again next year, good luck!"

#define UNIMPLEMENT2(_MSG_) Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "Sorry, I don't know how to '"<< _MSG_ <<"'. Try again next year, good luck!"

#define UNDEFINE_FUNCTION Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "This function is not defined!"

#define NOTHING_TODO Logger(LOG_VERBOSE)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "oh....... NOTHING TODO!"

#define DEADEND Logger(LOG_DEBUG)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
        << "WHAT YOU DO!! YOU SHOULD NOT GET HERE!!"

#define LOGGER Logger(LOG_LOG)

#define VERBOSE Logger(LOG_VERBOSE)

#define ERROR(_MSG_)  {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error(""));

#define RUNTIME_ERROR(_MSG_)  {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::runtime_error(""));

#define LOGIC_ERROR(_MSG_)  {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error(""));

#define OUT_RANGE_ERROR(_MSG_)  {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::out_of_range(""));

#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Logger(LOG_ERROR)<<__FILE__<<"["<<__LINE__<<"]:\n\t"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);

#define PARSER_ERROR(_MSG_)  {{ Logger(LOG_ERROR)<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"<<"\n\tConfigure fails :"<<(_MSG_) ;}throw(std::runtime_error(""));}

#include <cassert>
#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif

//#ifndef NDEBUG
#	define CHECK(_MSG_)    Logger(LOG_DEBUG) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< ( _MSG_)<<" "
//#else
//#	define CHECK(_MSG_)
//#endif

#define INFORM2(_MSG_) Logger(LOG_INFORM)<<__STRING(_MSG_)<<" = "<<_MSG_;

#define DOUBLELINE  std::setw(80) << std::setfill('=') << "="
//"--=============================================================--"
#define SINGLELINE  std::setw(80) << std::setfill('-') << "-"

#define SEPERATOR(_C_) std::setw(80) << std::setfill(_C_) << _C_
//"-----------------------------------------------------------------"

#define LOG_CMD(_CMD_) {auto __logger=Logger(LOG_LOG);__logger<<__STRING(_CMD_);_CMD_;__logger<<DONE;}

#define LOG_CMD2(_MSG_,_CMD_) {auto __logger=Logger(LOG_LOG);__logger<<_MSG_<<__STRING(_CMD_);_CMD_;__logger<<DONE;}

inline Logger & DONE(Logger & self)
{
	self.surffix("[DONE]");
	return self;
}

inline Logger & START(Logger & self)
{
	self.surffix("[START]");
	return self;
}

inline Logger & flush(Logger & self)
{
	self.flush();
	return self;
}

inline Logger & endl(Logger & self)
{
	self.endl();
	return self;
}
inline Logger & not_endl(Logger & self)
{
	self.not_endl();
	return self;
}
inline Logger & indent(Logger & self)
{
	self.SetIndent();
	return self;
}

inline Logger & TimeStamp(Logger & self)
{
	self << self.TimeStamp();
	return self;
}

struct SetLineWidth
{
	int width_;

	SetLineWidth(int width)
			: width_(width)
	{
	}
	~SetLineWidth()
	{
	}

};

inline LoggerStreams & operator<<(LoggerStreams & os, SetLineWidth const &setw)
{
	os.SetLineWidth(setw.width_);
	return os;

}

inline std::string ShowBit(unsigned long s)
{
	return std::bitset<64>(s).to_string();
}
#define CHECK_BIT(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0b"<< ShowBit( _MSG_)  << std::endl

#define CHECK_HEX(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0x"<<std::setw(20)<<std::setfill('0')<< std::hex<< ( _MSG_) << std::dec<< std::endl

//#define DONE    std::right<< " [Done]"
//#define START    std::right<<  " [START]"

}// namespace simpla
#endif /* LOG_H_ */

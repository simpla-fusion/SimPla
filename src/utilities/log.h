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

enum
{
	LOG_OUT_RANGE_ERROR = -4, LOG_LOGIC_ERROR = -3, LOG_ERROR = -2,

	LOG_WARNING = -1,

	LOG_INFORM = 0, LOG_LOG = 1, LOG_VERBOSE = 11, LOG_DEBUG = -1
};
class LoggerStreams: public SingletonHolder<LoggerStreams>
{
	size_t line_width_;
	size_t indent_;

public:
	static constexpr int DEFAULT_LINE_WIDTH = 100;
//	static constexpr char DEFAULT_FILENAME[] = "simpla.log";

	LoggerStreams(int l = LOG_LOG) :
			std_out_visable_level_(l), line_width_(DEFAULT_LINE_WIDTH), indent_(0)
	{
	}
	~LoggerStreams()
	{
		fs.close();
	}

	inline void OpenFile(std::string const & name)
	{
		if (fs.is_open())
			fs.close();

		fs.open(name.c_str(), std::ios_base::out);
	}

	void put(int level, std::string const & msg)
	{
		if (level <= std_out_visable_level_)
			std::cout << msg;

		if (fs.good())
		{
			fs << msg;
		}
		else
		{
			OpenFile("simpla.log");
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

public:
	typedef Logger this_type;

	Logger() :
			null_dump_(true), level_(0), current_line_char_count_(0), indent_(0)
	{
	}

	Logger(Logger const & r) :
			null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
					r.indent_)
	{
	}

	Logger(Logger && r) :
			null_dump_(r.null_dump_), level_(r.level_), current_line_char_count_(r.current_line_char_count_), indent_(
					r.indent_)
	{
	}

	Logger(int lv, size_t indent = 0) :
			null_dump_(false), level_(lv), current_line_char_count_(0), indent_(indent)
	{
		buffer_ << std::boolalpha;

		if (level_ == LOG_LOGIC_ERROR || level_ == LOG_ERROR || level_ == LOG_OUT_RANGE_ERROR)
		{
			buffer_ << "[E]";
		}
		else if (level_ == LOG_WARNING)
		{
			buffer_ << "[W]";
		}
		else if (level_ == LOG_LOG)
		{
			buffer_ << "[L]" << "[" << TimeStamp() << "]" << " ";
		}
		else if (level_ == LOG_VERBOSE)
		{
			buffer_ << "[V]" << "[" << TimeStamp() << "]" << " ";
		}
		else if (level_ == LOG_INFORM)
		{
		}
		else if (level_ == LOG_DEBUG)
		{
			buffer_ << "[D]";
		}

		size_t indent_width = LoggerStreams::instance().GetIndent();
		if (indent_width > 0)
			buffer_ << std::setfill('-') << std::setw(indent_width) << "+";

		SetIndent(indent_);

		current_line_char_count_ = GetBufferLength();
	}
	~Logger()
	{
		if (null_dump_)
			return;

		if (level_ == LOG_LOGIC_ERROR)
		{
			throw(std::logic_error(buffer_.str()));
		}
		else if (level_ == LOG_ERROR)
		{
			throw(std::runtime_error(buffer_.str()));
		}
		else if (level_ == LOG_OUT_RANGE_ERROR)
		{
			throw(std::out_of_range(buffer_.str()));
		}
		else
		{
			if (current_line_char_count_ > 0)
				buffer_ << std::endl;
			LoggerStreams::instance().put(level_, buffer_.str());
		}

		UnsetIndent(indent_);
	}

	void SetIndent(size_t n = 1)
	{
		LoggerStreams::instance().IncreaseIndent(n);
		indent_ += n;
	}
	void UnsetIndent(size_t n = 1)
	{
		if (indent_ >= n)
			LoggerStreams::instance().DecreaseIndent(n);
		indent_ -= n;
	}

	size_t GetBufferLength() const
	{
		return buffer_.str().size();
	}
	void flush()
	{
		LoggerStreams::instance().put(level_, buffer_.str());
		buffer_.str("");
	}
	void surffix(std::string const & s)
	{
		const_cast<this_type*>(this)->buffer_ << std::setfill('.')

		<< std::setw(LoggerStreams::instance().GetLineWidth() - current_line_char_count_)

		<< std::right << s << std::left;

		endl();

		flush();
	}

	void endl()
	{
		const_cast<this_type*>(this)->buffer_ << std::endl;
		current_line_char_count_ = 0;
	}

	template<typename T> inline this_type & operator<<(T const& value)
	{
		if (null_dump_)
			return *this;

		current_line_char_count_ -= GetBufferLength();

		const_cast<this_type*>(this)->buffer_ << value;

		current_line_char_count_ += GetBufferLength();

		if (current_line_char_count_ > LoggerStreams::instance().GetLineWidth())
			endl();

		return *this;
	}

	typedef Logger & (*LoggerStreamManipulator)(Logger &);

// take in a function with the custom signature
	Logger const& operator<<(LoggerStreamManipulator manip) const
	{
		if (null_dump_)
			return *this;
		// call the function, and return it's value
		return manip(*const_cast<this_type*>(this));
	}
	Logger & operator<<(LoggerStreamManipulator manip)
	{
		if (null_dump_)
			return *this;
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

	static void Verbose(int l = LOG_VERBOSE)
	{
		LoggerStreams::instance().SetStdOutVisableLevel(l);
	}

	static void OpenFile(std::string const & fname = "simpla_untitled.log")
	{
		LoggerStreams::instance().OpenFile(fname);
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

//FIXME The operator<< eat first input and transform to integral
#define ERROR Logger(LOG_ERROR)<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define LOGIC_ERROR Logger(LOG_LOGIC_ERROR)<<1<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define OUT_RANGE_ERROR Logger(LOG_OUT_RANGE_ERROR)<<1<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define WARNING Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define INFORM Logger(LOG_INFORM)

#define UNIMPLEMENT Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "This is a new year wish. Try again next year, good luck!"

#define UNIMPLEMENT2(_MSG_) Logger(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "Sorry, I don't know how to '"<< _MSG_ <<"'. Try again next year, good luck!"

#define NOTHING_TODO Logger(LOG_VERBOSE)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "NOTHING TODO"

#define DEADEND Logger(LOG_DEBUG)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
        << "WHAT YOU DO!! YOU SHOULD NOT GET HERE!!"

#define LOGGER Logger(LOG_LOG)

#define VERBOSE Logger(LOG_VERBOSE)

#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Logger(LOG_ERROR)<<__FILE__<<"["<<__LINE__<<"]:"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);
#include <cassert>
#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif

#ifndef NDEBUG
#	define CHECK(_MSG_)    Logger(LOG_DEBUG) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< ( _MSG_)
#	define EXCEPT(_COND_)    ((_COND_))?Logger():Logger(LOG_DEBUG)<<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_COND_)<<"="<< (_COND_)<<" "
#	define EXCEPT_EQ( actual,expected)    Logger(LOG_DEBUG,((expected)!=(actual) )) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(actual)<<" = "<< (actual) << " is not  "<< (expected) <<" "
#else
#	define CHECK(_MSG_)
#	define EXCEPT(_COND_)
#   define EXCEPT_EQ( actual,expected)
#endif

#define DOUBLELINE  std::setw(80) << std::setfill('=') << "="
//"--=============================================================--"
#define SINGLELINE  std::setw(80) << std::setfill('-') << "-"

#define SEPERATOR(_C_) std::setw(80) << std::setfill(_C_) << _C_
//"-----------------------------------------------------------------"

#define LOG_CMD(_CMD_) {auto __logger=Logger(LOG_LOG);__logger<<__STRING(_CMD_)<<flush;_CMD_;__logger<<DONE;}

inline Logger & DONE(Logger & self)
{
	//TODO: trigger timer
	self.surffix("[DONE]");
	return self;
}

inline Logger & START(Logger & self)
{
	//TODO: trigger timer
	self.surffix("[START]");
	return self;
}

inline Logger & flush(Logger & self)
{
	//TODO: trigger timer
	self.flush();
	return self;
}

inline Logger & endl(Logger & self)
{
	//TODO: trigger timer
	self.endl();
	return self;
}

inline Logger & indent(Logger & self)
{
	//TODO: trigger timer
	self.SetIndent();
	return self;
}

inline Logger & TimeStamp(Logger & self)
{
	//TODO: trigger timer
	self << self.TimeStamp();
	return self;
}
//#define DONE    std::right<< " [Done]"
//#define START    std::right<<  " [START]"

#endif /* LOG_H_ */

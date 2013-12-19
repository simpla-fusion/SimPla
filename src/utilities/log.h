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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

enum
{
	LOG_OUT_RANGE_ERROR = -4, LOG_LOGIC_ERROR = -3, LOG_ERROR = -2,

	LOG_WARNING = -1,

	LOG_INFORM = 0, LOG_LOG = 1, LOG_VERBOSE = 2, LOG_DEBUG = -1
};
class LoggerStreams: public SingletonHolder<LoggerStreams>
{
public:

	// TODO add multi_stream support

	LoggerStreams(int l = LOG_VERBOSE)
			: std_out_visable_level_(l)
	{
	}
	~LoggerStreams()
	{
		fs.close();
	}

	inline void OpenFile(std::string const & name)
	{
		if (fs.is_open())
		{
			fs.close();
		}

		fs.open(name.c_str(), std::ios_base::out);
	}
	void put(int level, std::string const & msg)
	{
		if (level <= std_out_visable_level_)
		{
			std::cout << msg;
		}
		if (fs.good())
		{
			fs << msg;
		}
	}
	inline void SetStdOutVisableLevel(int l)
	{
		std_out_visable_level_ = l;
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
	typedef std::ostringstream buffer_type;
	int level_;
	bool isVisable_;
	std::ostringstream *buffer_;
public:
	typedef Logger this_type;

	Logger(int lv = 0, bool cond = true)
			: level_(lv), isVisable_(cond), buffer_(new buffer_type)
	{

		if (level_ == LOG_LOGIC_ERROR || level_ == LOG_ERROR || level_ == LOG_OUT_RANGE_ERROR)
		{
			*static_cast<buffer_type*>(buffer_) << "[E]";
		}
		else if (level_ == LOG_WARNING)
		{
			*static_cast<buffer_type*>(buffer_) << "[W]";
		}
		else if (level_ == LOG_LOG)
		{
			*static_cast<buffer_type*>(buffer_) << "[L]" << "[" << TimeStamp() << "]" << " ";
		}
		else if (level_ == LOG_INFORM)
		{
		}
		else if (level_ == LOG_DEBUG)
		{
			*static_cast<buffer_type*>(buffer_) << "[D]";
		}

	}
	~Logger()
	{
		if (isVisable_)
		{

			*static_cast<buffer_type*>(buffer_) << std::endl;

			if (level_ == LOG_LOGIC_ERROR)
			{
				throw(std::logic_error(buffer_->str()));
			}
			else if (level_ == LOG_ERROR)
			{
				throw(std::runtime_error(buffer_->str()));
			}
			else if (level_ == LOG_OUT_RANGE_ERROR)
			{
				throw(std::out_of_range(buffer_->str()));
			}
			else
			{
				LoggerStreams::instance().put(level_, buffer_->str());
			}
		}

		delete buffer_;
	}

	template<typename T> inline this_type const & operator<<(T const & value) const
	{
		*static_cast<buffer_type*>(buffer_) << value;
		return *this;
	}

	this_type const & operator<<(bool value) const
	{
		*static_cast<buffer_type*>(buffer_) << std::boolalpha << value;
		return *static_cast<this_type const*>(this);
	}

	typedef this_type const& (*LoggerStreamManipulator)(this_type const&);

	// take in a function with the custom signature
	this_type const& operator<<(LoggerStreamManipulator manip) const
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
		manip(*static_cast<buffer_type*>(buffer_));
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
#	define EXCEPT(_COND_)    Logger(LOG_DEBUG,((_COND_)!=true)) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_COND_)<<"="<< (_COND_)<<" "
#	define EXCEPT_EQ( actual,expected)    Logger(LOG_DEBUG,((expected)!=(actual) )) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(actual)<<" = "<< (actual) << " is not  "<< (expected) <<" "
#else
#	define CHECK(_MSG_)
#	define EXCEPT(_COND_)
#endif
#define DOUBLELINE  std::setw(80) << std::setfill('=') << "="
//"--=============================================================--"
#define SINGLELINE  std::setw(80) << std::setfill('-') << "-"

#define SEPERATOR(_C_) std::setw(80) << std::setfill(_C_) << _C_
//"-----------------------------------------------------------------"

#endif /* LOG_H_ */

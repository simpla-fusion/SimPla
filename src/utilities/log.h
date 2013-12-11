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

	LOG_INFORM = 0, LOG_LOG = 1, LOG_VERBOSE = 2, LOG_DEBUG = 0
};
class LogStreams: public SingletonHolder<LogStreams>
{
public:

	// TODO add multi_stream support

	LogStreams(int l = LOG_VERBOSE) :
			std_out_visable_level_(l)
	{
	}
	~LogStreams()
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

class Log: public std::ostringstream
{
	int level_;
	bool isVisable_;
public:

	Log(int lv = 0, bool cond = true) :
			level_(lv), isVisable_(cond)
	{

		if (level_ == LOG_LOGIC_ERROR || level_ == LOG_ERROR
				|| level_ == LOG_OUT_RANGE_ERROR)
		{
			*this << "[E]";
		}
		else if (level_ == LOG_WARNING)
		{
			*this << "[W]";
		}
		else if (level_ == LOG_LOG)
		{
			*this << "[L]" << "[" << TimeStamp() << "]" << " ";
		}
		else if (level_ == LOG_INFORM)
		{
		}
		else if (level_ == LOG_DEBUG)
		{
			*this << "[D]";
		}

	}
	~Log()
	{
		if (isVisable_)
		{

			(*this) << std::endl;

			if (level_ == LOG_LOGIC_ERROR)
			{
				throw(std::logic_error(this->str()));
			}
			else if (level_ == LOG_ERROR)
			{
				throw(std::runtime_error(this->str()));
			}
			else if (level_ == LOG_OUT_RANGE_ERROR)
			{
				throw(std::out_of_range(this->str()));
			}
			else
			{
				LogStreams::instance().put(level_, this->str());
			}
		}
	}

	static void Verbose(int l = LOG_VERBOSE)
	{
		LogStreams::instance().SetStdOutVisableLevel(l);
	}

	static void OpenFile(std::string const & fname = "simpla_untitled.log")
	{
		LogStreams::instance().OpenFile(fname);
	}

	static std::string TimeStamp()
	{

		auto now = std::chrono::system_clock::to_time_t(
				std::chrono::system_clock::now());

		char mtstr[100];
		std::strftime(mtstr, 100, "%F %T", std::localtime(&now));

		return std::string(mtstr);
	}
private:
};

//FIXME The operator<< eat first input and transform to integral
#define ERROR Log(LOG_ERROR)<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define LOGIC_ERROR Log(LOG_LOGIC_ERROR)<<1<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define OUT_RANGE_ERROR Log(LOG_OUT_RANGE_ERROR)<<1<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define WARNING Log(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define INFORM Log(LOG_INFORM)

#define UNIMPLEMENT Log(LOG_WARNING)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
	          << "This is a new year wish. Try again next year, good luck!"

#define DEADEND Log(LOG_DEBUG)  <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:" \
        << "WHAT YOU DO!! YOU SHOULD NOT GET HERE!!"

#define LOG Log(LOG_LOG)

#define VERBOSE Log(LOG_VERBOSE)

#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Log(LOG_ERROR)<<__FILE__<<"["<<__LINE__<<"]:"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);
#include <cassert>
#ifdef NDEBUG
#  define ASSERT(_EXP_)
#else
#  define ASSERT(_COND_)    assert(_COND_);
#endif

#ifndef NDEBUG
#	define CHECK(_MSG_)    Log(LOG_DEBUG) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< ( _MSG_)
#	define EXCEPT(_COND_)    Log(LOG_DEBUG,((_COND_)!=true)) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_COND_)<<"="<< (_COND_)<<" "
#	define EXCEPT_EQ( actual,expected)    Log(LOG_DEBUG,((expected)!=(actual) )) <<" "<< (__FILE__) <<": line "<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(actual)<<" = "<< (actual) << " is not  "<< (expected) <<" "
#else
#	define CHECK(_MSG_)
#	define EXCEPT(_COND_)
#endif
#define DOUBLELINE "--=============================================================--"
#define SINGLELINE "-----------------------------------------------------------------"

#endif /* LOG_H_ */

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
#ifdef  OMP
#include <omp.h>
#endif
#include <time.h>
#define BOOST_DATE_TIME_POSIX_TIME_STD_CONFIG
#include <boost/date_time.hpp>

#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include "utilities/singleton_holder.h"
//#include <exception>
class LogStreams: public SingletonHolder<LogStreams>
{
public:

	// TODO add multi_stream support

	LogStreams() :
			info_level(0)
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
		if (level <= info_level)
		{
			std::cout << msg;
		}
		if (fs.good())
		{
			fs << msg;
		}
	}
	int info_level;

	std::string format;

	std::fstream fs;

};

class Log: public std::ostringstream
{
	int level_;
public:

	Log(int lv = 0) :
			level_(lv)
	{
		(*this)
#ifdef  _OMP
		<<"["<<omp_get_thread_num()<<"]"
#endif
		<< "[" << boost::posix_time::to_simple_string(clock_time()) << "]"
				<< " ";

	}
	~Log()
	{
		(*this) << std::endl;

		LogStreams::instance().put(level_, (*this).str());

		if (level_ == -3)
		{
			throw(std::logic_error(this->str()));
		}
		else if (level_ == -2)
		{
			throw(std::runtime_error(this->str()));
		}

	}

	static void Verbose(int l = 1)
	{
		LogStreams::instance().info_level = l;
	}

	static void OpenFile(std::string const & fname)
	{
		LogStreams::instance().OpenFile(fname);
	}

	static void setFormat(std::string const & format)
	{
		LogStreams::instance().format = format;
	}

	static std::string Teimstamp()
	{
		return boost::posix_time::to_simple_string(clock_time());
	}
private:
	static boost::posix_time::ptime clock_time()
	{
		timespec tv;
		clock_gettime(CLOCK_REALTIME, &tv);

		return (boost::posix_time::from_time_t(tv.tv_sec)
				+ boost::posix_time::nanosec(tv.tv_nsec));
	}
};
//FIXME The operator<< eat first input and transform to integral
#define ERROR Log(-2)<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define LOGIC_ERROR Log(-3)<<1<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define WARNING Log(-1)  <<"[W]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define INFORM Log(0)  <<"[I]"
//#ifdef DEBUG
//#define LOG Log(2) <<1<<"[L]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
//#else
#define LOG Log(2)  <<"[L]"
//#endif

#define VERBOSE Log(3)  <<"[V]"
<<<<<<< HEAD
=======
#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Log(-2)<<__FILE__<<"["<<__LINE__<<"]:"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);
>>>>>>> ddb1baf4864f73bec4047c704d79f5c9a1152544

#define CHECK(_MSG_)    Log(0) <<1 << (__FILE__) <<":"<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< _MSG_ <<std::endl

#define DOUBLELINE "================================================================="
#define SINGLELINE "-----------------------------------------------------------------"

#endif /* LOG_H_ */

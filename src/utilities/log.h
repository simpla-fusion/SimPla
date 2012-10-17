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
#include <boost/date_time/posix_time/posix_time.hpp>

#include <string>
#include <fstream>
#include <iostream>
#include <map>
//#include <exception>
namespace simpla
{
class Log: public std::ostringstream
{
	int level_;
public:
	static int info_level;

	Log(int l = 0) :
			level_(l)
	{
		(*this)
#ifdef  _OMP
		<<"["<<omp_get_thread_num()<<"]"
#endif
		<< "[" << boost::posix_time::to_simple_string(clock_time()) << "]";

	}
	~Log()
	{
		(*this) << std::endl;
		if (level_ <= info_level)
		{
			std::cout << (*this).str();
		}

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
		info_level = l;
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

#define ERROR Log(-2)<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define LOGIC_ERROR Log(-3)<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define WARNING Log(-1) <<"[W]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define INFORM Log(1) <<"[I]"
#define LOG Log(2) <<"[L]"
#define VERBOSE Log(3) <<"[V]"
#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Log(-2)<<__FILE__<<"["<<__LINE__<<"]:"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);

#define CHECK(_MSG_)    Log(0)  << (__FILE__) <<":"<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< _MSG_ <<std::endl

#define DOUBLELINE "================================================================="
#define SINGLELINE "-----------------------------------------------------------------"
} // namespace simpla

#endif /* LOG_H_ */

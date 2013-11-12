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

#include <utilities/singleton_holder.h>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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
	enum
	{
		L_LOGIC_ERROR = -3,
		L_ERROR = -2,
		L_WARNING = -1,
		L_INFORM = 0,
		L_LOG = 1,
		L_VERBOSE = 2
	};

	Log(int lv = 0) :
			level_(lv)
	{
		(*this)
//#ifdef  _OMP
//		<<"["<<omp_get_thread_num()<<"]"
//#endif
		<< "[" << TimeStamp() << "]" << " ";

	}
	~Log()
	{
		(*this) << std::endl;

		LogStreams::instance().put(level_, (*this).str());

		if (level_ == L_LOGIC_ERROR)
		{
			throw(std::logic_error(this->str()));
		}
		else if (level_ == L_ERROR)
		{
			throw(std::runtime_error(this->str()));
		}

	}

	static void Verbose(int l = L_INFORM)
	{
		LogStreams::instance().info_level = L_INFORM;
	}

	static void OpenFile(std::string const & fname)
	{
		LogStreams::instance().OpenFile(fname);
	}

	static void setFormat(std::string const & format)
	{
		LogStreams::instance().format = format;
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
#define ERROR Log(Log::L_ERROR)<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define LOGIC_ERROR Log(Log::L_LOGIC_ERROR)<<1<<"[E]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"

#define WARNING Log(Log::L_WARNING)  <<"[W]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
#define INFORM Log(Log::L_INFORM)  <<"[I]"
//#ifdef DEBUG
//#define LOG Log(2) <<1<<"[L]["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"
//#else
#define LOG Log(Log::L_LOG)  <<"[L]"
//#endif

#define VERBOSE Log(Log::L_VERBOSE)  <<"[V]"
//#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Log(-2)<<__FILE__<<"["<<__LINE__<<"]:"<< "Can not get enough memory! [ "  \
//        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);

#define CHECK(_MSG_)    Log(Log::L_INFORM) <<1 << (__FILE__) <<":"<< (__LINE__)<<":"<<  (__PRETTY_FUNCTION__) \
	<<"\n\t"<< __STRING(_MSG_)<<"="<< _MSG_ <<std::endl

#define DOUBLELINE "================================================================="
#define SINGLELINE "-----------------------------------------------------------------"

#endif /* LOG_H_ */

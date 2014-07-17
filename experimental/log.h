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
 *  created on: 2012-3-21
 *      Author: salmon
 */

#ifndef LOG_H_
#define LOG_H_

#include <sstream>
#include <bitset>
#include <complex>

#include "any.h"
namespace simpla
{

class LoggerStreams;

/**
 * \defgroup Logging Diagnostic logging features
 */
enum
{
	LOG_FORCE_OUTPUT = -10000,

	LOG_OUT_RANGE_ERROR = -4, LOG_LOGIC_ERROR = -3, LOG_ERROR = -2,

	LOG_WARNING = -1,

	LOG_INFORM = 0, LOG_LOG = 1, LOG_VERBOSE = 11, LOG_DEBUG = -20
};

/**
 *  \ingroup Logging
 *  \brief log message buffer,
 *
 *  @todo change to pimpl
 */
class Logger
{

public:
	typedef Logger this_type;

	Logger();

	Logger(Logger const & r);

	Logger(Logger && r);

	Logger(int lv, size_t indent = 0);

	~Logger();

	void init(int argc, char** argv);

	template<typename T> void set_property(std::string const & name, T const&v)
	{
		set_property_(name, Any(v));
	}

	template<typename T> T get_property(std::string const & name) const
	{
		return get_property_(name).template as<T>();
	}

	void set_indent(size_t n = 1);

	void unset_indent(size_t n = 1);

	size_t get_buffer_length() const;

	void flush();

	void surffix(std::string const & s);

	void endl();

	void not_endl();

	void time_stamp();

	template<typename T> inline Logger & operator<<(T const& value)
	{
		if (null_dump_)
			return *this;

		current_line_char_count_ -= get_buffer_length();

		const_cast<this_type*>(this)->buffer_ << value;

		current_line_char_count_ += get_buffer_length();

//		if (current_line_char_count_ > SingletonHolder<LoggerStreams>::instance().get_line_width())
//			endl();

		return *this;
	}

	typedef Logger & (*LoggerStreamManipulator)(Logger &);

	Logger & operator<<(LoggerStreamManipulator manip);

	typedef Logger & (*LoggerStreamConstManipulator)(Logger const &);

	// take in a function with the custom signature
	Logger const& operator<<(LoggerStreamConstManipulator manip) const;

	// this is the function signature of std::endl
	typedef std::basic_ostream<char, std::char_traits<char> > StdCoutType;
	typedef StdCoutType& (*StandardEndLine)(StdCoutType&);

	//! define an operator<< to take in std::endl
	this_type const& operator<<(StandardEndLine manip) const;
	this_type & operator<<(StandardEndLine manip);
private:

	bool null_dump_;
	typedef std::ostringstream buffer_type;
	int level_;
	std::ostringstream buffer_;
	size_t current_line_char_count_;
	size_t indent_;
	bool endl_;

	void set_property_(std::string const & name, Any const&);
	Any get_property_(std::string const & name) const;
};

/**
 * \ingroup Logging
 * \defgroup  logmanip  manip for Logger
 *
 * @{
 **/

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
	self.set_indent();
	return self;
}

inline Logger & TimeStamp(Logger & self)
{
	self.time_stamp();
	return self;
}

inline std::string ShowBit(unsigned long s)
{
	return std::bitset<64>(s).to_string();
}
/** @} */

/**
 *  \ingroup Logging
 *  \defgroup  LogShortCut    Shortcuts for logging
 *  @{
 */
#define LOG_STREAM Logger()

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

#define ERROR(_MSG_) { {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error("error"));}

#define RUNTIME_ERROR(_MSG_) { {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::runtime_error("runtime error"));}

#define LOGIC_ERROR(_MSG_)  {{Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::logic_error("logic error"));}

#define OUT_RANGE_ERROR(_MSG_) { {Logger(LOG_ERROR) <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:\n\t"<<(_MSG_);}throw(std::out_of_range("out of range"));}

#define ERROR_BAD_ALLOC_MEMORY(_SIZE_,_error_)    Logger(LOG_ERROR)<<__FILE__<<"["<<__LINE__<<"]:\n\t"<< "Can not get enough memory! [ "  \
        << _SIZE_ / 1024.0 / 1024.0 / 1024.0 << " GiB ]" << std::endl; throw(_error_);

#define PARSER_ERROR(_MSG_)  {{ Logger(LOG_ERROR)<<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"<<"\n\tConfigure fails :"<<(_MSG_) ;}throw(std::runtime_error(""));}

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

#define VERBOSE_CMD(_CMD_) {auto __logger=Logger(LOG_VERBOSE);__logger<<__STRING(_CMD_);_CMD_;__logger<<DONE;}

#define LOG_CMD1(_LEVEL_,_MSG_,_CMD_) {auto __logger=Logger(_LEVEL_);__logger<<_MSG_;_CMD_;__logger<<DONE;}

#define LOG_CMD2(_MSG_,_CMD_) {auto __logger=Logger(LOG_LOG);__logger<<_MSG_<<__STRING(_CMD_);_CMD_;__logger<<DONE;}

#define CHECK_BIT(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0b"<< ShowBit( _MSG_)  << std::endl

#define CHECK_HEX(_MSG_)  std::cout<<std::setfill(' ')<<std::setw(30) <<__STRING(_MSG_)<<" = 0x"<<std::setw(20)<<std::setfill('0')<< std::hex<< ( _MSG_) << std::dec<< std::endl

//#define DONE    std::right<< " [Done]"
//#define START    std::right<<  " [START]"

/** @} */

}// namespace simpla
#endif /* LOG_H_ */

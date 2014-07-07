/**
 * \file simpla_defs.h
 *
 *    \date 2011-12-24
 *    \author  salmon
 */

#ifndef SIMPLA_DEFS_H_
#define SIMPLA_DEFS_H_

#include <string>
#include "utilities/log.h"

namespace simpla
{

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif
#define SIMPLA_LOGO      "\n"                                  \
" ____ ___ __  __ ____  _       \n"\
"/ ___|_ _|  \\/  |  _ \\| | __ _ \n"\
"\\___ \\| || |\\/| | |_) | |/ _` |\n"\
" ___) | || |  | |  __/| | (_| |\n"\
"|____/___|_|  |_|_|   |_|\\__,_|\n"\
"\n"                                  \
" SimPla, Plasma Simulator        \n"                         \
" Build Date: " __DATE__ " " __TIME__"                   \n"\
" ID:" IDENTIFY "                                        \n"\
" Author:  YU Zhi. All rights reserved.           \n"

inline std::string ShowShortVersion()
{
	return IDENTIFY;
}
inline std::string ShowVersion()
{
	return SIMPLA_LOGO;
}
inline std::string ShowCopyRight()
{
	return SIMPLA_LOGO;
}

inline void TheStart(int flag = 1)
{
	switch (flag)
	{
	default:
		INFORM << SINGLELINE;
		VERBOSE << "So far so good, let's start work! ";
		INFORM << "[MISSOIN     START]: " << TimeStamp;
		INFORM << SINGLELINE;
	}
}
inline void TheEnd(int flag = 1)
{
	switch (flag)
	{
	case -2:
		INFORM << "Oop! Some thing wrong! Don't worry, maybe not your fault!\n"
				" Just maybe! Please Check your configure file again! ";
		break;
	case -1:
		INFORM << "Sorry! I can't help you now! Please, Try again later!";
		break;
	case 0:
		INFORM << "See you! ";
		break;
	case 1:
	default:
		LOGGER << "MISSION COMPLETED!";

		INFORM << SINGLELINE;
		INFORM << "[MISSION COMPLETED]: " << TimeStamp;
		VERBOSE << "Job is Done!! ";
		VERBOSE << "	I'm so GOOD!";
		VERBOSE << "		Thanks me please!";
		VERBOSE << "			Thanks me please!";
		VERBOSE << "You are welcome!";
		INFORM << SINGLELINE;

	}
	LOGGER << std::endl;
	INFORM << std::endl;

	exit(0);
}

}
/**
 *  \include design_doc.txt
 */
#endif /* SIMPLA_DEFS_H_ */

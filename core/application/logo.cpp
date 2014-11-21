/*
 * logo.cpp
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#include "logo.h"

#include <cstdlib>
#include <iomanip>
#include <string>

#include "../utilities/log.h"

namespace simpla
{

#ifndef IDENTIFY
#	define IDENTIFY "UNKNOWN"
#endif

#define AUTHOR " YU Zhi <yuzhi@ipp.ac.cn> "

#ifndef COPYRIGHT
#	define COPYRIGHT  " All rights reserved."
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
" ID:" IDENTIFY "\n"  \
" Author: " AUTHOR "\n"  \
COPYRIGHT

std::string ShowLogo()
{
	return SIMPLA_LOGO;
}

std::string ShowVersion()
{
	return IDENTIFY;
}
std::string ShowCopyRight()
{
	return SIMPLA_LOGO;
}

void TheStart(int flag)
{
	switch (flag)
	{
	default:
		INFORM << SINGLELINE;
		VERBOSE << "So far so good, let's start work! ";
		INFORM << "[MISSOIN     START]: ";
		INFORM << SINGLELINE;
	}
}
void TheEnd(int flag)
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
		break;
	case 1:
	default:
		LOGGER << "MISSION COMPLETED!";

		INFORM << SINGLELINE;
		INFORM << "[MISSION COMPLETED]: ";
		VERBOSE << "Job is Done!! ";
		VERBOSE << "	I'm so GOOD!";
		VERBOSE << "		Thanks me please!";
		VERBOSE << "			Thanks me please!";
		VERBOSE << "You are welcome!";
		INFORM << SINGLELINE;

	}
	exit(0);
}
}
 // namespace simpla

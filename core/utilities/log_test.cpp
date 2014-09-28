/*
 * log.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "log.h"
using namespace simpla;
int main(int argc, char **argv)
{
	Logger::init(argc, argv);

	int a;

	LOG_CMD(a = 5);

	Logger::set_stdout_visable_level(10);

	auto logger = LOGGER;

	logger << "Hello world " << flush

	<< __STRING(a=5) << "lalalalal";

	a = 6;

	logger << DONE

	<< a << std::endl;

	INFORM << "What?";
	INFORM << "Why?";
	INFORM << "Who?";
	INFORM << "WHen?";
}

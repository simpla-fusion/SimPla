/*
 * log.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../log.h"
using namespace simpla;
int main(int argc, char **argv)
{
	logger::Logger::init(argc, argv);

	int a;

	LOG_CMD(a = 5);

//	log::Logger::set_message_visable_level(10);

	auto L = LOGGER;

	L << "Hello world " << std::endl

	<< logger::flush

	<< __STRING(a=5) << "lalalalal";

	a = 6;

	L << DONE<< a << std::endl;

	INFORM<< "What?"<< std::endl;
	INFORM<< "Why?"<< std::endl;
	INFORM<< "Who?"<< std::endl;
	INFORM<< "WHen?"<< std::endl;
}

/*
 * log.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../utilities/log.h"

using namespace simpla;

int main(int argc, char **argv)
{
    logger::init(argc, argv);

    logger::set_stdout_level(29);
    int a;

    LOG_CMD(a = 5);

//	log::Logger::set_message_visable_level(10);

    logger::Logger L(logger::LOG_LOG);

    L << "Hello world " << std::endl

    << logger::flush

    << __STRING(a = 5) << "lalalalal";

    a = 6;

    L << DONE << a << std::endl;

    INFORM << "What?" << std::endl;
    INFORM << "Why?" << std::endl;
    INFORM << "Who?" << std::endl;
    INFORM << "WHen?" << __PRETTY_FUNCTION__ << "[" << __FILE__ << "]" << std::endl;
}

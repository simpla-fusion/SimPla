/*
 * log.cpp
 *
 *  Created on: 2013年12月28日
 *      Author: salmon
 */

#include "log.h"

int main(int argc, char **argv)
{
	int a;

	LOG_CMD(a = 5);

	auto logger = LOGGER;

	logger << "Hello world " << flush

	<< __STRING(a=5) << "lalalalal";

	a = 6;

	logger << DONE

	<< a;
}

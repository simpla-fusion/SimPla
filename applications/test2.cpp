/*
 * test2.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <iostream>
#include "../src/utilities/log.h"

int main(int argc, char **argv)
{
	long s = 16 | 32 | 2;

	CHECK(((((s) & 3) + 1) % 3 - 1));
	CHECK(((((s >> 2) & 3) + 1) % 3 - 1));
	CHECK(((((s >> 4) & 3) + 1) % 3 - 1));
}

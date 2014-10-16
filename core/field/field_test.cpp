/*
 * field_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */
#include <iostream>
#include "field.h"
#include "../parallel/block_range.h"
using namespace simpla;

int main(int argc, char **argv)
{
	BlockRange<size_t> domain(0, 10);

	Field<BlockRange<size_t>, double> f(domain);

	std::cout << f.size() << std::endl;
	f = 1.23456;

	for (int i = 0; i < 10; ++i)
	{
		std::cout << f[i] << std::endl;
	}

}

/*
 * sparse_vector_test.cpp
 *
 *  Created on: 2013年7月28日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include "equation.h"

using namespace simpla;

int main(int argc, char **argv)
{

	std::map<size_t, double> m;

	PlaceHolder x1(14);
	PlaceHolder x2(15);

	(-(x1 + x2) / 3.5 + 5 + 4.25 * (-x1 * 3 - x2 * 2) * 2).assign(m, 1);

	for (int i = 0; i < 10; ++i)
	{
		PlaceHolder(i, i * 3).assign(m, 1);
	}
	for (auto it = m.begin(); it != m.end(); ++it)
	{
		std::cout << "[" << (it->first) << "," << (it->second) << "]"
				<< std::endl;
	}
}


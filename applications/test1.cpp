/*
 * test1.cpp
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#include <chrono>
#include <ratio>
#include <thread>
#include <vector>

#include "../src/utilities/log.h"

inline signed int Shift(unsigned int d, int s)
{
	constexpr int m = 4;
	constexpr int n = sizeof(signed int) / sizeof(char) * 8;

	return ((static_cast<signed int>(d) << (n - m * (s + 1))) >> (n - m));

}
int main()
{
	for (int i = 0; i < 256; ++i)
	{
		std::cout << "[" << std::setw(4) << std::setfill('0') << i << " = "

		<< std::setw(3) << std::hex << i << " ]--> "

		<< std::dec << std::setfill(' ')

		<< std::setw(5) << Shift(i, 0)

		<< std::setw(5) << Shift(i, 1)

		<< std::setw(5) << Shift(i, 2)

		<< std::endl;
	}
}


/*
 * test_nullptr.cpp
 *
 *  Created on: 2013年12月26日
 *      Author: salmon
 */
#include <iostream>
int main(int argc, char **argv)
{

	int * a = nullptr;

	a[0] = 2;
	std::cout << a[0] << std::endl;

}


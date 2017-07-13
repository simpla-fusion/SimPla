/**
 * \file properties_test.cpp
 *
 * @date    2014-7-13  AM8:06:32
 * @author salmon
 */
#include <gtest/gtest.h"
#include <iostream>
#include "properties.h"
using namespace simpla;

TEST(properties,set)
{
	Properties prop;

	prop["first"] = 3.141923;
	std::cout << prop["first"].as<double>() << std::endl;
	prop["first"] = 123445UL;
	std::cout << prop["first"].as<unsigned long>() << std::endl;

	prop["first"]["Sub1"] = 188;
	prop["first"]["Sub2"] = 266;
	prop["first"]["Sub3"] = 699;
	for (auto const & item : prop["first"])
	{
		std::cout << item.first << "=" << item.second.as<int>() << std::endl;

		std::cout << std::boolalpha << item.second["what"].empty() << std::endl;
	}

}

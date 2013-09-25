/*
 * lua_parser_test.cpp
 *
 *  Created on: 2013年9月24日
 *      Author: salmon
 */

#include "lua_parser.h"
#include <iostream>
using namespace simpla;
int main(int argc, char** argv)
{
	LuaObject lstate;

	lstate.ParseString(
			"c=100 \n t1={a=5,b=6.0,c=\"text\"} \n t2={e=4,f=true} \n\n"
					"function f(x,y) \n"
					"    return x+y  \n"
					"end \n");

	lstate.ParseString(argv[1]);

	std::cout << "c \t="<< lstate["c"].as<double>() << std::endl;

	std::cout << "c \t="<< lstate["c"].as<double>() << std::endl;

	std::cout << "c \t="<< lstate["c"].as<double>() << std::endl;

	std::cout << "t1.a \t="<< lstate["t1"]["a"].as<double>() << std::endl;

	std::cout << "t2.f \t="<<std::boolalpha << lstate["t2"]["f"].as<bool>() << std::endl;

	std::cout << "f(2,2.5) \t="<< lstate["f"](2.0, 2.5).as<double>() << std::endl;

	std::cout << "f(1.2,2.5) \t="<< lstate["f"](1.2, 2.5).as<double>() << std::endl;

	std::cout << "f(3,2.5) \t="<< lstate["f"](3.0, 2.5).as<double>() << std::endl;

	std::cout << "f(3,2.5) \t="<< lstate["f"](3.0, 2.5).as<double>() << std::endl;

	std::cout << "f(3,2.5) \t="<< lstate["f"](3.0, 2.5).as<double>() << std::endl;



}


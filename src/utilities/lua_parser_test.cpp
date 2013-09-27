/*
 * lua_parser_test.cpp
 *
 *  Created on: 2013年9月24日
 *      Author: salmon
 */

#include "lua_parser.h"
#include "fetl/primitives.h"
#include <iostream>
#include "utilities/parse_config.h"
using namespace simpla;
int main(int argc, char** argv)
{
	typedef ParseConfig<LuaObject> Parser;
	Parser * pt = new Parser();

	pt->ParseString(
			"c=100 \n t1={a=5,b=6.0,c=\"text\",e={a=5,b=6.0}} \n t2={e=4,f=true} \n t3={1,3,4,5}\n"
					"function f(x,y) \n"
					"    return x+y  \n"
					"end \n");

	if (argc > 1)
	{
		pt->ParseString(argv[1]);
	}
	std::cout << "c \t=" << pt->Get<double>("c") << std::endl;

	std::cout << "c \t=" << pt->Get<double>("c") << std::endl;

	std::cout << "c \t=" << pt->Get<double>("c") << std::endl;

	std::cout << "t1.a \t=" << (*pt)["t1"].Get<double>("a") << std::endl;

	std::cout << "t2.f \t=" << std::boolalpha << (*pt)["t2"].Get<double>("f")
			<< std::endl;

	double res;

	(*pt)["f"].Function(&res, 2.0, 2.5);

	std::cout << "f(2,2.5) \t=" << res << std::endl;

	(*pt)["f"].Function(&res, 1.2, 2.5);

	std::cout << "f(1.2,2.5) \t=" << res << std::endl;

	(*pt)["f"].Function(&res, 3, 2.5);

	std::cout << "f(3,2.5) \t=" << res << std::endl;

	(*pt)["f"].Function(&res, 3, 2.5);

	std::cout << "f(3,2.5) \t=" << res << std::endl;

	(*pt)["f"].Function(&res, 3, 2.5);

	std::cout << "f(3,2.5) \t=" << res << std::endl;

	std::cout << "t3 \t=" << pt->Get<nTuple<3, double>>("t3") << std::endl;

	LuaObject * t1 = new LuaObject((*pt)["t1"]);

	LuaObject * e = new LuaObject((*t1)["e"]);

	delete pt;

	delete t1;

	std::cout << "t1.e.a \t= " << e->Get<double>("a") << std::endl;

	delete e;

}


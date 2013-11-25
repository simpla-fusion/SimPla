/*
 * lua_parser_test.cpp
 *
 *  Created on: 2013年9月24日
 *      Author: salmon
 */

#include "lua_state.h"
#include "fetl/primitives.h"
#include <iostream>
#include <map>
using namespace simpla;
int main(int argc, char** argv)
{
	LuaObject pt;

	if (argc > 1)
	{
		pt.ParseFile(argv[1]);
	}
	else
	{
		pt.ParseString(
				"c=100 \n t1={a=5,b=6.0,c=\"text\",e={a=5,b=6.0}} \n t2={e=4,f=true} \n t3={1,3,4,5}\n"
						"tt={e=12, d=13,h=2} \n"
						"function f(x,y) \n"
						"    return x+y  \n"
						"end \n");
	}

	std::cout << "c \t=" << pt.at("c").as<double>() << std::endl;

	std::cout << "c \t=" << pt["c"].as<int>() << std::endl;

	std::cout << "c \t=" << pt["c"].as<double>() << std::endl;

	std::cout << "b \t=" << pt.Get<int>("b", 120) << std::endl;
//
//	auto tt1 = pt.at("t1");
//
////	std::cout << "t1 \t=" << tt1.as<double>() << std::endl;
//
////	std::cout << "t2.f \t=" << std::boolalpha
////			<< pt.GetChild("t2")["f"].as<bool>() << std::endl;
//
	std::cout << "f(2,2.5) \t=" << pt["f"](2.0, 2.5).as<double>() << std::endl;

	std::cout << "f(1.2,2.5) \t=" << pt["f"](1.2, 2.5).as<double>()
			<< std::endl;

	std::cout << "f(3,2.5) \t=" << pt["f"](3.0, 2.5).as<double>() << std::endl;

	std::cout << "f(3,2.5) \t=" << pt["f"](2.0, 2.5).as<double>() << std::endl;

	std::cout << "t3 \t=" << pt.GetChild("t3").as<nTuple<3, double>>()
			<< std::endl;

	LuaObject * t1 = new LuaObject(pt["t1"]);

	LuaObject * e = new LuaObject(t1->GetChild("e"));

	delete t1;

	std::cout << "t1.e.a \t= " << e->GetChild("a").as<double>() << std::endl;

	delete e;
//
////	auto it = pt.GetChild("tt").begin();
////	std::cout << (*it).first.as<std::string>() << std::endl;
////	std::cout << (*it).second.as<int>() << std::endl;
////	++it;
////	std::cout << (*it).second.as<int>() << std::endl;
////	++it;
////	std::cout << (*it).second.as<int>() << std::endl;
//
//	std::cout << pt.GetChild("tt")[1].as(10) << std::endl;
//
//	pt.GetChild("tt").ForEach(
//
//	[&](LuaObject const& key,LuaObject const&value)
//	{
//		std::cout << key.as<std::string>("")
//		<<" = "<< value.as<int>() << std::endl;
//	});
//
//	for (auto const & p : pt.GetChild("tt"))
//	{
//		std::cout << p.first.as<std::string>("") << " = " << p.second.as<int>()
//				<< std::endl;
//	}
}


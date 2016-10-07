/*
 * lua_parser_test.cpp
 *
 *  created on: 2013-9-24
 *      Author: salmon
 */

#include <iostream>
#include <string>
#include <utility>

#include "../LuaObject.h"
#include "../nTuple.h"
#include "../nTupleExt.h"

using namespace simpla;

int main(int argc, char **argv)
{
    toolbox::LuaObject pt;
    pt.init();

    pt.parse_string(
            "c=100 \n t1={a=5,b=6.0,c={\"text\",\"text2\"},e={a=5},f={2,3,4}} \n t2={e=4,f=true} \n t3={{2,3,4},{6,7,9}}\n"

                    "tt={6,6,7,3,e=12, d=13,h=2} \n"
                    "function f(x,y) \n"
                    "    return (x-y)   \n"
                    "end \n"
                    "tuple={123,456}"
                    "ntuple={1,2,3}"
                    "");

//    if (argc > 1) { pt.parse_file(argv[1]); }

    {
        Properties props;

        pt["t1"].as(&props);
        std::cout << props << std::endl;
        std::cout << "a=" << props["a"].as<double>() << std::endl;
        std::cout << "f=" << props["f"].as<nTuple<double, 3>>() << std::endl;
        std::cout << "f=" << props["f"].as<nTuple<int, 3>>() << std::endl;

    }
    {
        Properties props;

        pt["t3"].as(&props);
//        std::cout << props << std::endl;
        std::cout << "t3=" << props.template as<Matrix<double, 2, 3> >() << std::endl;

    }
//    for (int i = 0; i < 10; ++i)
//    {
//        auto l = pt["f"];
//
//        l(1.0, 2.0).as<int>();
//    }
//
//
//    std::cout << "c \t=" << pt["c"].as<int>() << std::endl;
//
//    std::cout << "c \t=" << pt["c"].as<double>() << std::endl;
////
//	auto tt1 = pt.at("t1");
//
////	std::cout << "t1 \t=" << tt1.as<double>() << std::endl;
//
////	std::cout << "t2.f \t=" << std::boolalpha
////			<< pt.get_child("t2")["f"].as<bool>() << std::endl;
//
//    std::cout << "f(2,2.5) \t=" << pt["f"](2.0, 2.5).as<double>() << std::endl;
//
//    std::cout << "f(1.2,2.5) \t=" << pt["f"](1.2, 2.5).as<double>()
//    << std::endl;
//
//    std::cout << "f(3,2.5) \t=" << pt["f"](3.0, 2.5).as<double>() << std::endl;
////
//    std::cout << "f(3,2.5) \t=" << pt["f"](2.0, 2.5).as<double>() << std::endl;
//
//	for (int i = 0; i < 10; ++i)
//	{
//
//		std::cout << "t3 \t=" << pt.get_child("t3").as<nTuple<double, 3>>()
//				<< std::endl;
////	}
//
//    lua::LuaObject *t1 = new lua::LuaObject(pt["t1"]);
//
//    lua::LuaObject *e = new lua::LuaObject(t1->get_child("e"));
//
//    delete t1;
//
//    std::cout << "t1.e.a \t= " << e->get_child("a").as<double>() << std::endl;
//
//    delete e;
//
//    for (auto it = pt.get_child("tt").begin(), e = pt.get_child("tt").end();
//         it != e; ++it)
//    {
//        std::cout << (*it).first.as<std::string>() << " = "
//        << (*it).second.as<int>() << std::endl;
//
//    }
//    std::cout << "============================" << std::endl;
//    auto pp = pt.get_child("tt");
//
//    for (auto it = pp.begin(), e = pp.end(); it != e; ++it)
//    {
//        std::cout << (*it).first.as<std::string>() << " = "
//        << (*it).second.as<int>() << std::endl;
//
//    }
//    std::cout << "============================" << std::endl;
//    pt.parse_string("tt.e=1000");
//
//    for (auto it = pp.begin(), e = pp.end(); it != e; ++it)
//    {
//        std::cout << (*it).first.as<std::string>() << " = "
//        << (*it).second.as<int>() << std::endl;
//
//    }
//    std::cout << "============================" << std::endl;
//
//
//
//    //	pt.get_child("tt").ForEach(
////
////	[&](lua::GeoObject const& key,lua::GeoObject const&value)
////	{
////		std::cout << key.as<std::string>()
////		<<" = "<< value.as<int>() << std::endl;
////	}
////
////	);
////	auto obj = pt.get_child("tt");
////	size_t num = obj.GetLength();
////	for (size_t i = 0; i < num; ++i)
////	{
////		std::cout << obj[i].as<int>() << std::endl;
////	}
////
//    for (auto const &p : pt.get_child("tt"))
//    {
//        std::cout << p.first.as<std::string>() << " = " << p.second.as<int>()
//        << std::endl;
//    }
//    for (auto const &p : pt.get_child("tt"))
//    {
//        std::cout << p.first.as<std::string>() << " = " << p.second.as<int>()
//        << std::endl;
//    }
//    for (auto const &p : pt.get_child("tt"))
//    {
//        std::cout << p.first.as<std::string>() << " = " << p.second.as<int>()
//        << std::endl;
//    }
//	auto fobj = pt["f"];
//
//	std::cout << "(1.23340 - 2.4560 )*2= "
//			<< pt["f"](std::make_tuple(1.23340, 2.4560, 2)).as<double>()
//			<< std::endl;
//
//	double a, b;
//	std::tie(a, b) = pt["tuple"].as_tuple<double, double>();
//	std::cout << " a= " << a << "  b= " << b << std::endl;
//
//	nTuple<double, 3> c = { 4, 5, 6 };
//	std::cout << c << std::endl;
//
//	c = pt["ntuple"].as<nTuple<double, 3>>();
//
//	std::cout << c << std::endl;

//	std::vector<double> d;
//	pt["ntuple"].as(&d);
//	std::cout << d << std::endl;
}


/**
 * @file demo_misc3.cpp
 *
 * @date 2015年4月16日
 * @author salmon
 */

#include "../../core/utilities/utilities.h"
using namespace simpla;

int main(int argc, char **argv)
{
	LuaObject lobj;
	lobj.parse_string("a=103");

	std::cout << static_cast<int>(lobj["a"]) << std::endl;
	std::cout << static_cast<double>(lobj["a"]) << std::endl;
//	std::cout << static_cast<std::string>(lobj["a"]) << std::endl;

}

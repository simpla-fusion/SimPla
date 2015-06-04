/*
 * geqdsk_convert.cpp
 *
 *  created on: 2013-12-3
 *      Author: salmon
 */

#include <memory>
#include <string>
#include "geqdsk.h"
using namespace simpla;
int main(int argc, char ** argv)
{
	GEqdsk geqdsk((std::string(argv[1])));

	geqdsk.print(std::cout);
	geqdsk.Write(argv[1]);

//	GLOBAL_DATA_STREAM.open_file("geqdsk_test");
//	GLOBAL_DATA_STREAM.cd("/");
//	geqdsk.save();
}

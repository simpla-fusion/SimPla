/*
 * geqdsk_convert.cpp
 *
 *  Created on: 2013-12-3
 *      Author: salmon
 */

#include <memory>
#include <string>
#include "geqdsk.h"
using namespace simpla;
int main(int argc, char ** argv)
{
	GEqdsk geqdsk((std::string(argv[1])));

	geqdsk.Print(std::cout);
	geqdsk.Write(argv[1]);

//	GLOBAL_DATA_STREAM.OpenFile("geqdsk_test");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	geqdsk.Save();
}

/*
 * geqdsk_convert.cpp
 *
 *  Created on: 2013年12月3日
 *      Author: salmon
 */

#include <memory>
#include <string>
#include "geqdsk.h"
using namespace simpla;
int main(int argc, char ** argv)
{
	GEqdsk geqdsk(argv[1]);

	geqdsk.Print(std::cout);
	geqdsk.Write(argv[1], DataStream::XDMF);

//	GLOBAL_DATA_STREAM.OpenFile("geqdsk_test");
//	GLOBAL_DATA_STREAM.OpenGroup("/");
//	geqdsk.Save();
}

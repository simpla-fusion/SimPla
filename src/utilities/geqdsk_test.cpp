/*
 * geqdsk_test.cpp
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

	if (argc > 2)
		geqdsk.ReadProfile(argv[2], 4);
	geqdsk.Print(std::cout);
	geqdsk.Write(argv[1], GEqdsk::XDMF);


	GLOBAL_DATA_STREAM.OpenFile("geqdsk_test");
	GLOBAL_DATA_STREAM.OpenGroup("/");
	geqdsk.Save();
}

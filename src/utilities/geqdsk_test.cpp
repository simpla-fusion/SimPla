/*
 * geqdsk_test.cpp
 *
 *  Created on: 2013年12月3日
 *      Author: salmon
 */

#include "geqdsk.h"
#include <memory>
#include <string>

using namespace simpla;
int main(int argc, char ** argv)
{
	GEqdsk geqdsk(argv[1]);
//	geqdsk.Print(std::cout);
	geqdsk.Write(argv[1], GEqdsk::XDMF);
}

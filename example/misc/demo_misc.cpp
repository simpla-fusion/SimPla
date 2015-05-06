/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include "../../core/mesh/mesh_ids.h"
#include "../../core/utilities/utilities.h"

using namespace simpla;

int main(int argc, char **argv)
{
	SHOW(sizeof(unsigned long));
	SHOW(sizeof(std::int32_t));
	SHOW(sizeof(std::int64_t));
	SHOW(sizeof(MeshIDs::id_s));

	MeshIDs::id_s t = { 1, 0xF2, 2, 4 };

	SHOW(std::is_pod<MeshIDs::id_s>::value);

	SHOW_HEX(static_cast<std::uint64_t>(t));

	SHOW(t.i);
	SHOW(t.j);
	SHOW(t.k);
	SHOW(t.h);
	SHOW((static_cast<nTuple<long, 4> >(t)));

//
//	std::cout << std::hex << MeshIDs::_DI << std::endl;
//
//	std::cout << std::hex << MeshIDs::_DJ << std::endl;
//
//	std::cout << std::hex << MeshIDs::_DK << std::endl;
}


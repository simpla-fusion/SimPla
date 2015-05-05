/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */

#include <cstdint>
#include <iostream>
#include <limits>

#include "../../core/gtl/ntuple.h"
#include "../../core/utilities/utilities.h"
#include "../../core/mesh/mesh_ids.h"

using namespace simpla;

int main(int argc, char **argv)
{

//	SHOW(std::numeric_limits<std::int16_t>::max());
//	std::cout << std::hex << v << std::endl;
//	std::cout << std::hex << raw_cast<id_tuple>(v)[0] << std::endl;
//	std::cout << std::hex << raw_cast<id_tuple>(v)[1] << std::endl;
//	std::cout << std::hex << raw_cast<id_tuple>(v)[2] << std::endl;
//	std::cout << std::hex << raw_cast<id_tuple>(v)[3] << std::endl;
//
//	std::cout << std::hex << static_cast<id_type>(u) << std::endl;

	auto s = MeshIDs::pack(nTuple<int, 3>( { 1, 2, 3 }));

	std::cout << std::hex << s << std::dec << std::endl;

	std::cout << MeshIDs::unpack(s) << std::endl;

	std::cout << MeshIDs::unpack(s - MeshIDs::_DI) << std::endl;
	std::cout << MeshIDs::unpack(s - MeshIDs::_DJ) << std::endl;
	std::cout << MeshIDs::unpack(s - MeshIDs::_DK) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DI) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DJ) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DK) << std::endl;
	std::cout << MeshIDs::coordinates(s) << std::endl;

	s = MeshIDs::pack_diff(nTuple<int, 3>( { 1, 2, 3 }));

	std::cout << MeshIDs::unpack_diff(s) << std::endl;
	std::cout << MeshIDs::unpack(s - MeshIDs::_DI) << std::endl;
	std::cout << MeshIDs::unpack(s - MeshIDs::_DJ) << std::endl;
	std::cout << MeshIDs::unpack(s - MeshIDs::_DK) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DI) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DJ) << std::endl;
	std::cout << MeshIDs::unpack(s + MeshIDs::_DK) << std::endl;
	std::cout << MeshIDs::coordinates(s) << std::endl;
}

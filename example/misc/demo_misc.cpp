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
	typename MeshIDs::id_type a = 0; //MeshIDs::_DA << 2UL;
	typename MeshIDs::id_type b = MeshIDs::_DA;

	std::cout
			<< MeshIDs::unpack_index(MeshIDs::diff(a, b))[0]
					- static_cast<long>(MeshIDs::INDEX_ZERO >> 3UL)
			<< std::endl;

	std::cout
			<< MeshIDs::unpack_index(MeshIDs::diff(b, a))[0]
					- static_cast<long>(MeshIDs::INDEX_ZERO >> 3UL)
			<< std::endl;
	std::cout
			<< static_cast<MeshIDs::index_type>(MeshIDs::unpack_index(
					MeshIDs::diff(a, b))[0]) << std::endl;
	CHECK_BIT(MeshIDs::OVERFLOW_FLAG);
	CHECK_BIT(MeshIDs::INDEX_ZERO);
	CHECK_BIT(MeshIDs::diff(a, b));

	CHECK_BIT(raw_cast<unsigned long>(-1L));
	CHECK_BIT((-1L));



}


/**
 * @file demo_misc2.cpp
 *
 * @date 2015-4-16
 * @author salmon
 */

#include "../../core/gtl/ntuple.h"
#include "utilities.h"
#include "../../core/mesh/mesh_ids.h"

#include <iostream>

using namespace simpla;

int main(int argc, char **argv)
{
	MeshIDs::id_type node_ids[] = { 0, 1, 6 };
	for (auto n_id : node_ids)
	{
		MeshIDs::id_type b = MeshIDs::pack_index(
				(nTuple<size_t, 3> { 0, 0, 0 }), n_id);

		MeshIDs::id_type e = MeshIDs::pack_index(
				(nTuple<size_t, 3> { 6, 1, 1 }), n_id);

		MeshIDs::range_type r1(b, e);

		size_t count = 0;
		for (auto s : r1)
		{
			SHOW_HEX((MeshIDs::template type_cast<nTuple<size_t, 4> >(s)));
			SHOW(MeshIDs::hash(s, b, e));
			SHOW(MeshIDs::sub_index(s));
			++count;
		}

		SHOW("=====================");
	}

}

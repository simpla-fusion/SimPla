/*
 * mesh.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef MESH_H_
#define MESH_H_

#include <fetl/ntuple.h>
#include <fetl/primitives.h>
#include <cstddef>
#include <vector>

namespace simpla
{
/**
 * @brief  Uniform interface to all mesh.
 * @ingroup mesh
 */
template<typename _Mesh>
struct MeshTraits
{
	static const int NUM_OF_DIMS = 3;

	template<typename Element> using Container=std::vector<Element>;

	typedef size_t index_type;

	typedef size_t size_type;

	typedef nTuple<NUM_OF_DIMS, Real> coordinates_type;

	void Init();

	size_type get_num_elements(int iform);

	coordinates_type get_coordinates(int iform, index_type const &);

	void get_coordinates(int iform, int num, index_type const idxs[],
			coordinates_type *x);

	size_t get_cell_num_vertices(int iform, index_type const &idx);

	void get_cell_vertices(int iform, index_type const &idx, index_type idxs[]);

	index_type get_cell_index(coordinates_type const &);



};

} //namespace simpla

#endif /* MESH_H_ */

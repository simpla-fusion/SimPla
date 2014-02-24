/*
 * topology_rect.cpp
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#include "topology_rect.h"

#include <functional>
#include <iomanip>
#include <iostream>

#include "../fetl/ntuple.h"
#include "../utilities/utilities.h"

namespace simpla
{

TopologyRect::TopologyRect()
		: tags_(*this)
{
}

TopologyRect::~TopologyRect()
{
}

void TopologyRect::Update()
{

	//configure

	num_cells_ = 1;
	num_grid_points_ = 1;

	for (int i = 0; i < NUM_OF_DIMS; ++i)
	{
		if (dims_[i] <= 1)
		{
			dims_[i] = 1;

			dims_[i] = 1;

			ghost_width_[i] = 0;

		}
		else
		{
			num_cells_ *= dims_[i];
			num_grid_points_ *= dims_[i];

		}
	}

	// Fast first
#ifndef ARRAY_C_ORDER

	strides_[2] = dims_[1] * dims_[0];
	strides_[1] = dims_[0];
	strides_[0] = 1;

#else
	strides_[2] = 1;
	strides_[1] = dims_[2];
	strides_[0] = dims_[1] * dims_[2];
#endif

	for (int i = 0; i < NUM_OF_DIMS; ++i)
	{
		if (dims_[i] <= 1)
			strides_[i] = 0;
	}

	coordinates_shift_[0][0][0] = 0.0;
	coordinates_shift_[0][0][1] = 0.0;
	coordinates_shift_[0][0][2] = 0.0;

	coordinates_shift_[3][0][0] = 0.0;
	coordinates_shift_[3][0][1] = 0.0;
	coordinates_shift_[3][0][2] = 0.0;

	coordinates_shift_[1][0][0] = 0.5;
	coordinates_shift_[1][0][1] = 0.0;
	coordinates_shift_[1][0][2] = 0.0;

	coordinates_shift_[1][1][0] = 0.0;
	coordinates_shift_[1][1][1] = 0.5;
	coordinates_shift_[1][1][2] = 0.0;

	coordinates_shift_[1][2][0] = 0.0;
	coordinates_shift_[1][2][1] = 0.0;
	coordinates_shift_[1][2][2] = 0.5;

	coordinates_shift_[2][0][0] = 0.0;
	coordinates_shift_[2][0][1] = 0.5;
	coordinates_shift_[2][0][2] = 0.5;

	coordinates_shift_[2][1][0] = 0.5;
	coordinates_shift_[2][1][1] = 0.0;
	coordinates_shift_[2][1][2] = 0.5;

	coordinates_shift_[2][2][0] = 0.5;
	coordinates_shift_[2][2][1] = 0.5;
	coordinates_shift_[2][2][2] = 0.0;

}

std::ostream &
TopologyRect::Save(std::ostream &os) const
{

	os

	<< "{" << "\n"

	<< std::setw(10) << "Topology" << " = { \n "

	<< "\t" << std::setw(10) << "Type" << " = \"" << GetTopologyTypeAsString() << "\", \n"

	<< "\t" << std::setw(10) << "Dimensions" << " = {" << ToString(dims_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "GhostWidth" << " = {" << ToString(ghost_width_, ",") << "}, \n "

	<< std::setw(10) << "}, \n "

	<< "} \n ";

	return os;
}

std::ostream &
operator<<(std::ostream & os, TopologyRect const & d)
{
	d.Save(os);
	return os;
}

void TopologyRect::_Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
        std::function<void(index_type)> const &fun) const
{

	index_type ib = IX * dims_[0] * thread_id / num_threads;
	index_type ie = IX * dims_[0] * (thread_id + 1) / num_threads;

	index_type jb = 0;
	index_type je = IY * dims_[1];

	index_type kb = 0;
	index_type ke = IZ * dims_[2];

	int mb = 0;
	int me = num_comps_per_cell_[IFORM];

	for (index_type i = ib; i < ie; i += IX)
		for (index_type j = jb; j < je; j += IY)
			for (index_type k = kb; k < ke; k += IZ)
			{
				for (int m = mb; m < me; ++m)
				{
					fun(i + j + k + PutM(m));
				}
			}

}
}  // namespace simpla

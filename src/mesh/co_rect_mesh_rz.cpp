/*
 * co_rect_mesh.cpp
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */
#include "co_rect_mesh_rz.h"
namespace simpla
{

CoRectMeshRZ::CoRectMeshRZ()
		: tags_(*this)
{
}

CoRectMeshRZ::~CoRectMeshRZ()
{
}

void CoRectMeshRZ::Update()
{

	//configure

	num_cells_ = 1;
	num_grid_points_ = 1;
	cell_volume_ = 1.0;
	d_cell_volume_ = 1.0;
	for (int i = 0; i < NUM_OF_DIMS; ++i)
	{
		if (dims_[i] <= 1)
		{
			dims_[i] = 1;
			dx_[i] = 0.0;
			inv_dx_[i] = 0.0;

			dS_[0][i] = 0.0;

			dS_[1][i] = 0.0;

			dims_[i] = 1;

			ghost_width_[i] = 0;

		}
		else
		{
			dx_[i] = (xmax_[i] - xmin_[i]) / static_cast<Real>(dims_[i]);

			if (std::abs(xmax_[i] - xmin_[i]) > std::numeric_limits<Real>::epsilon())
			{
				inv_dx_[i] = 1.0 / dx_[i];
				cell_volume_ *= dx_[i];
				dS_[0][i] = 1.0 / dx_[i];
				dS_[1][i] = -1.0 / dx_[i];
				d_cell_volume_ /= dx_[i];
			}
			else
			{
				inv_dx_[i] = 0.0;
				dS_[0][i] = 0.0;
				dS_[1][i] = 0.0;
			}

			num_cells_ *= (dims_[i]);

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

	coordinates_shift_[1][0][0] = 0.5 * dx_[0];
	coordinates_shift_[1][0][1] = 0.0;
	coordinates_shift_[1][0][2] = 0.0;

	coordinates_shift_[1][1][0] = 0.0;
	coordinates_shift_[1][1][1] = 0.5 * dx_[1];
	coordinates_shift_[1][1][2] = 0.0;

	coordinates_shift_[1][2][0] = 0.0;
	coordinates_shift_[1][2][1] = 0.0;
	coordinates_shift_[1][2][2] = 0.5 * dx_[2];

	coordinates_shift_[2][0][0] = 0.0;
	coordinates_shift_[2][0][1] = 0.5 * dx_[1];
	coordinates_shift_[2][0][2] = 0.5 * dx_[2];

	coordinates_shift_[2][1][0] = 0.5 * dx_[0];
	coordinates_shift_[2][1][1] = 0.0;
	coordinates_shift_[2][1][2] = 0.5 * dx_[2];

	coordinates_shift_[2][2][0] = 0.5 * dx_[0];
	coordinates_shift_[2][2][1] = 0.5 * dx_[1];
	coordinates_shift_[2][2][2] = 0.0;

}


void CoRectMeshRZ::_Traversal(unsigned int num_threads, unsigned int thread_id, int IFORM,
        std::function<void(int, index_type, index_type, index_type)> const &fun) const
{

//	index_type ib = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[0];
//	index_type ie = ((flags & WITH_GHOSTS) > 0) ? dims_[0] : dims_[0] - ghost_width_[0];
//
//	index_type jb = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[1];
//	index_type je = ((flags & WITH_GHOSTS) > 0) ? dims_[1] : dims_[1] - ghost_width_[1];
//
//	index_type kb = ((flags & WITH_GHOSTS) > 0) ? 0 : ghost_width_[2];
//	index_type ke = ((flags & WITH_GHOSTS) > 0) ? dims_[2] : dims_[2] - ghost_width_[2];

	index_type ib = 0;
	index_type ie = dims_[0];

	index_type jb = 0;
	index_type je = dims_[1];

	index_type kb = 0;
	index_type ke = dims_[2];

	int mb = 0;
	int me = num_comps_per_cell_[IFORM];

	index_type len = ie - ib;
	index_type tb = ib + len * thread_id / num_threads;
	index_type te = ib + len * (thread_id + 1) / num_threads;

	for (index_type i = tb; i < te; ++i)
		for (index_type j = jb; j < je; ++j)
			for (index_type k = kb; k < ke; ++k)
				for (int m = mb; m < me; ++m)
				{
					fun(m, i, j, k);
				}

}

  std::ostream & operator<<(std::ostream & os, CoRectMeshRZ const & d)
{
	d.Save(os);
	return os;
}

std::ostream & CoRectMeshRZ::Save(std::ostream &os) const
{

	os

	<< "{" << "\n"

	<< std::setw(10) << "Type" << " = \"" << GetTypeName() << "\", \n"

	<< std::setw(10) << "Topology" << " = { \n "

	<< "\t" << std::setw(10) << "Type" << " = \"" << GetTopologyTypeAsString() << "\", \n"

	<< "\t" << std::setw(10) << "Dimensions" << " = {" << ToString(dims_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "GhostWidth" << " = {" << ToString(ghost_width_, ",") << "}, \n "

	<< std::setw(10) << "}, \n "

	<< std::setw(10) << "Geometry" << " = { \n "

	<< "\t" << std::setw(10) << "Type" << " = \"Origin_DxDyDz\", \n "

	<< "\t" << std::setw(10) << "Origin" << " = {" << ToString(xmin_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "DxDyDz" << " = {" << ToString(dx_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "Min" << " = {" << ToString(xmin_, ",") << "}, \n "

	<< "\t" << std::setw(10) << "Max" << " = {" << ToString(xmax_, ",") << "}, \n "

	<< std::setw(10) << "}, \n "

	<< std::setw(10) << "dt" << " = " << GetDt() << ",\n"

	<< std::setw(10) << "Unit System" << " = " << constants() << ",\n"

	<< std::setw(10) << "Media Tags" << " = " << tags() << "\n"

	<< "} \n ";

	return os;
}
}  // namespace simpla


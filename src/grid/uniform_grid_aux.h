/*
 * uniform_grid_aux.h
 *
 *  Created on: 2013-6-24
 *      Author: salmon
 */

#ifndef UNIFORM_GRID_AUX_H_
#define UNIFORM_GRID_AUX_H_

#include "utilities/properties.h"
namespace simpla {

inline void Parse(ptree const &pt)
{
	boost::optional<std::string> ot = pt.get_optional < std::string
			> ("Topology.<xmlattr>.Type");
	if (!ot || *ot != "CoRectMesh")
	{
		ERROR << "Grid type mismatch";
	}

	dims = pt.get < IVec3 > ("Topology.<xmlattr>.Dimensions");

	gw = pt.get < IVec3 > ("Topology.<xmlattr>.Ghostwidth");

	xmin = pt.get < Vec3 > ("Geometry.XMin");

	xmax = pt.get < Vec3 > ("Geometry.XMax");

	dt = pt.get("Time.<xmlattr>.dt", 1.0d);

	for (int i = 0; i < NDIMS; ++i)
	{
		gw[i] = (gw[i] * 2 > dims[i]) ? dims[i] / 2 : gw[i];
		if (dims[i] <= 1)
		{
			dims[i] = 1;
			xmax[i] = xmin[i];
			dx[i] = 0.0;
			inv_dx[i] = 0.0;
		}
		else
		{
			dx[i] = (xmax[i] - xmin[i]) / static_cast<Real>(dims[i] - 1);
			inv_dx[i] = 1.0 / dx[i];
		}
	}

	strides[2] = 1;
	strides[1] = dims[2];
	strides[0] = dims[1] * dims[2];

//#pragma omp parallel for  << here can not be parallized
	for (Index i = 0; i < dims[0]; ++i)
		for (Index j = 0; j < dims[1]; ++j)
			for (Index k = 0; k < dims[2]; ++k)
			{
				Index s = (i * strides[0] + j * strides[1] + k * strides[2]);

				for (int f = 0; f < 4; ++f)
				{
					Index num_of_comp = get_num_of_comp(f);
					for (Index l = 0; l < num_of_comp; ++l)
					{
						if (i < gw[0] || i > dims[0] - gw[0] //
						|| j < gw[1] || j > dims[1] - gw[1] //
						|| k < gw[2] || k > dims[2] - gw[2])
						{
							ghost_ele_[f].push_back(s * num_of_comp + l);
						}
						else
						{

							center_ele_[f].push_back(s * num_of_comp + l);
						}
					}

				}
			}

}

inline std::string Summary() const
{
	std::ostringstream os;

	os

	<< std::setw(20) << "Grid Type : " << "UniformRect" << std::endl

	<< std::setw(20) << "Grid dims : " << dims << std::endl

	<< std::setw(20) << "Range : " << xmin << " ~ " << xmax << std::endl

	<< std::setw(20) << "dx : " << dx << std::endl

	<< std::setw(20) << "dt : " << dt << std::endl;

	return os.str();
}


}  // namespace simpla

#endif /* UNIFORM_GRID_AUX_H_ */

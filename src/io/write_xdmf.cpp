/*
 * wrte_xdmf.cpp
 *
 *  Created on: 2012-9-30
 *      Author: salmon
 */

//#include "write_xdmf.h"

#include <cstddef>
#include <sstream>
#include <string>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../mesh/uniform_rect.h"

namespace simpla
{

std::string CreateFileTemplate(CoRectMesh const & mesh)
{
	std::ostringstream ss;
	ss << "<?xml version='1.0' ?>" << std::endl

	<< "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>" << std::endl

	<< "<Xdmf Version='2.0'>" << std::endl

	<< "<Domain>" << std::endl

	<< "<Grid Name='%GRID_NAME%' GridType='Uniform' >" << std::endl;

	int ndims = 0;
	Real xmin[3], dx[3];
	size_t dims[3];
	for (int i = 0; i < 3; ++i)
	{
		if (mesh.dims_[i] > 1)
		{
			xmin[ndims] = mesh.xmin_[i];
			dx[ndims] = mesh.dx_[i];
			dims[ndims] = mesh.dims_[i];
			++ndims;
		}
	}

	if (ndims == 3) //3D
	{

		ss << "  <Topology TopologyType='3DCoRectMesh'  "

		<< " Dimensions='" << dims[0] << " " << dims[1] << " " << dims[2]

		<< "'></Topology>" << std::endl

		<< "  <Geometry Type='Origin_DxDyDz'>" << std::endl

		<< "    <DataItem Format='XML' Dimensions='3'>"

		<< xmin[0] << " " << xmin[1] << " " << xmin[2] << "</DataItem>"

		<< std::endl

		<< "    <DataItem Format='XML' Dimensions='3'>"

		<< dx[0] << " " << dx[1] << " " << dx[2] << "</DataItem>" << std::endl

		<< "  </Geometry>" << std::endl;

	}
	else if (ndims == 2) //2D
	{
		ss

		<< "  <Topology TopologyType='2DCoRectMesh'  "

		<< " Dimensions='"

		<< dims[0] << " " << dims[1] << "'></Topology>" << std::endl

		<< "  <Geometry Type='Origin_DxDy'>" << std::endl

		<< "    <DataItem Format='XML' Dimensions='2'>"

		<< xmin[0] << " " << xmin[1] << "</DataItem>" << std::endl

		<< "    <DataItem Format='XML' Dimensions='2'>"

		<< dx[0] << " " << dx[1] << "</DataItem>" << std::endl

		<< "  </Geometry>" << std::endl;
	}

	ss << "<!-- ADD_ATTRIBUTE_HERE -->" << std::endl

	<< "</Grid>" << std::endl

	<< "</Domain>" << std::endl

	<< "</Xdmf>" << std::endl;

	return ss.str();
}

} // namespace simpla

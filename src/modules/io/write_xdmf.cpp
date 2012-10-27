/*
 * wrte_xdmf.cpp
 *
 *  Created on: 2012-9-30
 *      Author: salmon
 */

#include "write_xdmf.h"
#include <algorithm>
#include <string>
#include <list>
#include <H5Cpp.h>
#include <hdf5_hl.h>
#include <fstream>
#include <sstream>
#include "fetl/grid/uniform_rect.h"
#include "engine/context.h"
#include "engine/modules.h"
#include "engine/object.h"
#include "write_hdf5.h"

namespace simpla
{
namespace io
{
template<>
WriteXDMF<UniformRectGrid>::~WriteXDMF()
{
}
template<>
void WriteXDMF<UniformRectGrid>::Initialize()

{

	std::ostringstream ss;
	ss << "<?xml version='1.0' ?>" << std::endl

	<< "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>" << std::endl

	<< "<Xdmf Version='2.0'>" << std::endl

	<< "<Domain>" << std::endl;

	int ndims = 0;
	Real xmin[3], dx[3];
	size_t dims[3];
	for (int i = 0; i < 3; ++i)
	{
		if (grid.dims[i] > 1)
		{
			xmin[ndims] = grid.xmin[i];
			dx[ndims] = grid.dx[i];
			dims[ndims] = grid.dims[i];
			++ndims;
		}
	}

	if (ndims == 3) //3D
	{

		ss << "<Grid Name='GRID" << ctx.Counter() << "' GridType='Uniform' >"
				<< std::endl

				<< "  <Topology TopologyType='3DCoRectMesh'  "

				<< " Dimensions='" << dims[0] << " " << dims[1] << " "
				<< dims[2] << "'></Topology>" << std::endl

				<< "  <Geometry Type='Origin_DxDyDz'>" << std::endl

				<< "    <DataItem Format='XML' Dimensions='3'>"

				<< xmin[0] << " " << xmin[1] << " " << xmin[2] << "</DataItem>"
				<< std::endl

				<< "    <DataItem Format='XML' Dimensions='3'>"

				<< dx[0] << " " << dx[1] << " " << dx[2] << "</DataItem>"
				<< std::endl

				<< "  </Geometry>" << std::endl;

	}
	else if (ndims == 2) //2D
	{
		ss << "<Grid Name='GRID" << ctx.Counter() << "' GridType='Uniform' >"
				<< std::endl

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

	ss

	<< attrPlaceHolder << std::endl

	<< "</Grid>" << std::endl

	<< "</Domain>" << std::endl

	<< "</Xdmf>" << std::endl;

	file_template = ss.str();

	mkdir(path_.c_str(), 0777);

	LOG << "Create module WriteXDMF";

}
template<>
void WriteXDMF<UniformRectGrid>::Eval()
{
	LOG << "Run module WriteXDMF";

	std::string filename;

	{
		std::ostringstream st;
		st << std::setw(8) << std::setfill('0') << ctx.Counter();
		filename = st.str();
	}

	H5::Group grp = H5::H5File(path_ + "/" + filename + ".h5", H5F_ACC_TRUNC) //
	.openGroup("/");

	std::string xmdf_file(file_template);

	{
		std::ostringstream ss;
		ss << "  <Time  Value='" << ctx.Timer() << "' />" << std::endl;
		xmdf_file.insert(xmdf_file.find(attrPlaceHolder, 0), ss.str());
	}

	for (std::list<std::string>::const_iterator it = obj_list_.begin();
			it != obj_list_.end(); ++it)
	{
		std::map<std::string, TR1::shared_ptr<Object> >::const_iterator oit =
				ctx.objects.find(*it);

		if (oit != ctx.objects.end())
		{

			int ndim = oit->second->get_dimensions();

			size_t dims[ndim];

			oit->second->get_dimensions(dims);

			int pndim = 0;
			size_t pdims[ndim];
			{

				for (int i = 0; i < ndim; ++i)
				{
					if (dims[i] > 1)
					{
						pdims[pndim] = dims[i];
						++pndim;
					}
				}
			}
			std::ostringstream ss;

			ss << "  <Attribute Name='" << (*it)
					<< "'  AttributeType='Vector' Center='Node' >" << std::endl

					<< "    <DataItem  NumberType='Float' Precision='8' Format='HDF' Dimensions='";

			for (int i = 0; i < pndim; ++i)
			{
				ss << pdims[i] << " ";
			}

			ss << "' >" << std::endl

			<< filename << ".h5:/" << (*it)

			<< "    </DataItem>" << std::endl

			<< "  </Attribute>" << std::endl;

			xmdf_file.insert(xmdf_file.find(attrPlaceHolder, 0), ss.str());

			WriteHDF5(grp, (*it), *(oit->second));
		}
	}

	std::fstream fs((path_ + "/" + filename + ".xdmf").c_str(),
			std::fstream::out);
	fs << xmdf_file;
	fs.close();
}

} // namespace io
} // namespace simpla

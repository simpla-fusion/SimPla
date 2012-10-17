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
#include <boost/algorithm/string.hpp>
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
WriteXDMF<UniformRectGrid>::WriteXDMF(Context<UniformRectGrid> const & d,
		const ptree & pt) :
		ctx(d),

		grid(d.grid),

		attrPlaceHolder("<!-- Add Attribute Here -->"),

		step(pt.get("step", 1))
{

	BOOST_FOREACH(const ptree::value_type &v, pt.get_child("Objects"))
	{
		std::string id = v.second.get_value<std::string>();
		boost::algorithm::trim(id);
		obj_list_.push_back(id);
	}

	std::ostringstream ss;
	ss << "<?xml version='1.0' ?>" << std::endl

	<< "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>" << std::endl

	<< "<Xdmf Version='2.0'>" << std::endl

	<< "<Domain>" << std::endl;

	if (grid.NDIMS == 3) //3D
	{

		ss << "<Grid Name='GRID" << ctx.Counter() << "' GridType='Uniform' >"
				<< std::endl

				<< "  <Topology TopologyType='3DCoRectMesh'  "

				<< " Dimensions='" << grid.dims << "'></Topology>" << std::endl

				<< "  <Geometry Type='Origin_DxDyDz'>" << std::endl

				<< "    <DataItem Format='XML' Dimensions='3'>"

				<< grid.xmin << "</DataItem>" << std::endl

				<< "    <DataItem Format='XML' Dimensions='3'>"

				<< grid.dx << "</DataItem>" << std::endl

				<< "  </Geometry>" << std::endl;

	}
	else if (grid.NDIMS == 2) //2D
	{
		nTuple<TWO, size_t> dims =
		{ grid.dims[0], grid.dims[1] };
		nTuple<TWO, Real> xmin =
		{ grid.xmin[0], grid.xmin[1] };
		nTuple<TWO, Real> dx =
		{ grid.dx[0], grid.dx[1] };

		ss << "<Grid Name='GRID" << ctx.Counter() << "' GridType='Uniform' >"
				<< std::endl

				<< "  <Topology TopologyType='2DCoRectMesh'  "

				<< " Dimensions='" << dims << "'></Topology>" << std::endl

				<< "  <Geometry Type='Origin_DxDyDz'>" << std::endl

				<< "    <DataItem Format='XML' Dimensions='2'>"

				<< xmin << "</DataItem>" << std::endl

				<< "    <DataItem Format='XML' Dimensions='2'>"

				<< dx << "</DataItem>" << std::endl

				<< "  </Geometry>" << std::endl;
	}

	ss << "  <Time  Value='" << ctx.Timer() << "' />" << std::endl

	<< attrPlaceHolder << std::endl

	<< "</Grid>" << std::endl

	<< "</Domain>" << std::endl

	<< "</Xdmf>" << std::endl;

	file_template = ss.str();

	dir_path_ = pt.get("DirPath", "Untitled");

	mkdir(dir_path_.c_str(), 0777);

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

	H5::Group grp = H5::H5File(dir_path_ + "/" + filename + ".h5",
			H5F_ACC_TRUNC) //
	.openGroup("/");

	std::string xmdf_file(file_template);

	for (std::list<std::string>::const_iterator it = obj_list_.begin();
			it != obj_list_.end(); ++it)
	{
		std::map<std::string, TR1::shared_ptr<Object> >::const_iterator oit =
				ctx.objects.find(*it);

		CHECK(*it);

		if (oit != ctx.objects.end())
		{
			CHECK(*it+"is found!");

			int ndim = oit->second->get_dimensions();

			size_t dims[ndim];

			oit->second->get_dimensions(dims);

			std::ostringstream ss;

			ss << "  <Attribute Name='" << (*it)
					<< "'  AttributeType='Vector' Center='Node' >" << std::endl

					<< "    <DataItem  NumberType='Float' Precision='8' Format='HDF' Dimensions='";

			for (int i = 0; i < ndim; ++i)
			{
				ss << dims[i] << " ";
			}

			ss << "' >" << std::endl

			<< filename << ".h5:/" << (*it)

			<< "    </DataItem>" << std::endl

			<< "  </Attribute>" << std::endl;

			xmdf_file.insert(xmdf_file.find(attrPlaceHolder, 0), ss.str());

			WriteHDF5(grp, (*it), *(oit->second));
		}
	}

	std::fstream fs((dir_path_ + "/" + filename + ".xdmf").c_str(),
			std::fstream::out);
	fs << xmdf_file;
	fs.close();
}

} // namespace io
} // namespace simpla

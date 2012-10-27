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
#include "engine/object.h"
#include "fetl/fetl.h"

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

	<< "<Domain>" << std::endl

	<< "<Grid Name='GRID" << ctx.Counter() << "' GridType='Uniform' >"
			<< std::endl;

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

	ss << attrPlaceHolder << std::endl

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
			Object & obj = *oit->second;

			std::string attr_str = "Scalar";

			H5::DataType mdatatype(
					H5LTtext_to_dtype(obj.get_element_type_desc().c_str(),
							H5LT_DDL));

			int mem_nd = 0;
			hsize_t mdims[MAX_XDMF_NDIMS];

			if (obj.CheckType(typeid(Field<Grid, IZeroForm, Real> ))
					|| obj.CheckType(typeid(Field<Grid, IZeroForm, Complex> )))
			{
			}
			else if (obj.CheckType(typeid(Field<Grid, IOneForm, Real> ))
					|| obj.CheckType(typeid(Field<Grid, ITwoForm, Real> ))
					|| obj.CheckType(typeid(Field<Grid, IOneForm, Complex> ))
					|| obj.CheckType(typeid(Field<Grid, ITwoForm, Complex> )))
			{
				attr_str = "Vector";
				mem_nd = 1;
				mdims[0] = THREE;
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Real> > )))
			{
				attr_str = "Vector";

				mdatatype = H5::PredType::NATIVE_DOUBLE;
				mem_nd = 1;
				mdims[0] = THREE;
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Complex> > )))
			{
				attr_str = "Vector";
				mdatatype = H5LTtext_to_dtype(
						DataType<Complex>().desc().c_str(), H5LT_DDL);
				mem_nd = 1;
				mdims[0] = THREE;
			}

			for (int i = 0; i < THREE; ++i)
			{
				if (grid.dims[i] > 1)
				{
					mdims[mem_nd] = grid.dims[i];
					++mem_nd;
				}
			}

			std::ostringstream ss;

			ss << "  <Attribute Name='" << (*it) << "'  AttributeType='"
					<< attr_str << "' Center='Node' >" << std::endl

					<< "    <DataItem  NumberType='Float' Precision='8' Format='HDF' Dimensions='";

			for (int i = 0; i < mem_nd; ++i)
			{
				ss << mdims[i] << " ";
			}

			ss << "' >" << std::endl

			<< filename << ".h5:/" << (*it)

			<< "    </DataItem>" << std::endl

			<< "  </Attribute>" << std::endl;

			xmdf_file.insert(xmdf_file.find(attrPlaceHolder, 0), ss.str());

			H5::DataSet dataset = grp.createDataSet((*it).c_str(), mdatatype,
					H5::DataSpace(mem_nd, mdims));

			dataset.write(obj.get_data(), mdatatype);

		}
	}

	std::fstream fs((path_ + "/" + filename + ".xdmf").c_str(),
			std::fstream::out);
	fs << xmdf_file;
	fs.close();
}

} // namespace io
} // namespace simpla

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
#include "engine/arrayobject.h"
#include "fetl/fetl.h"

namespace simpla
{
namespace io
{

template<>
WriteXDMF<UniformRectGrid>::WriteXDMF(Context<UniformRectGrid> const & d,
		const ptree & pt) :
		ctx(d),

		grid(d.grid),

		file_template(""),

		attrPlaceHolder("<!-- Add Attribute Here -->"),

		stride_(pt.get("<xmlattr>.Stride", 1))

{
	BOOST_FOREACH(const typename ptree::value_type &v, pt)
	{
		std::string id = v.second.get_value<std::string>();
		boost::algorithm::trim(id);
		obj_list_.push_back(id);
	}

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

	LOG << "Create module WriteXDMF";

	mkdir(ctx.output_path.c_str(), 0777);
}

template<>
WriteXDMF<UniformRectGrid>::~WriteXDMF()
{
}
template<>
void WriteXDMF<UniformRectGrid>::Eval()
{
	if (ctx.Counter() % stride_ != 0)
	{
		return;
	}

	LOG << "Run module WriteXDMF";

	std::string filename;

	{
		std::ostringstream st;
		st << std::setw(8) << std::setfill('0') << ctx.Counter();
		filename = st.str();
	}


	H5::Group grp = H5::H5File(ctx.output_path + "/" + filename + ".h5",
			H5F_ACC_TRUNC).openGroup("/");

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
			ArrayObject & obj = *TR1::dynamic_pointer_cast<ArrayObject>(
					oit->second);

			std::string attr_str = "Scalar";

			H5::DataType mdatatype(
					H5LTtext_to_dtype(obj.get_element_type_desc().c_str(),
							H5LT_DDL));

			int nd = 0;
			hsize_t xdmf_dims[MAX_XDMF_NDIMS];

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
				nd = 1;
				xdmf_dims[0] = THREE;
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Real> > )))
			{
				attr_str = "Vector";

				mdatatype = H5::PredType::NATIVE_DOUBLE;
				nd = 1;
				xdmf_dims[0] = THREE;
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Complex> > )))
			{
				attr_str = "Vector";
				mdatatype = H5LTtext_to_dtype(
						DataType<Complex>().desc().c_str(), H5LT_DDL);
				nd = 1;
				xdmf_dims[0] = THREE;
			}

			for (int i = 0; i < THREE; ++i)
			{
				if (grid.dims[i] > 1)
				{
					xdmf_dims[nd] = grid.dims[i];
					++nd;
				}
			}

			std::ostringstream ss;

			ss << "  <Attribute Name='" << (*it) << "'  AttributeType='"
					<< attr_str << "' Center='Node' >" << std::endl
					<< "    <DataItem  NumberType='Float' Precision='8' Format='HDF' Dimensions='";

			for (int i = 0; i < nd; ++i)
			{
				ss << xdmf_dims[i] << " ";
			}

			ss << "' >" << std::endl

			<< filename << ".h5:/" << (*it)

			<< "    </DataItem>" << std::endl

			<< "  </Attribute>" << std::endl;

			xmdf_file.insert(xmdf_file.find(attrPlaceHolder, 0), ss.str());

			hsize_t h5_dims[MAX_XDMF_NDIMS];
			for (int i = 0; i < nd; ++i)
			{
				h5_dims[i] = xdmf_dims[nd - 1 - i];
			}

			H5::DataSet dataset = grp.createDataSet((*it).c_str(), mdatatype,
					H5::DataSpace(nd, h5_dims));

			dataset.write(obj.get_data(), mdatatype);

		}
	}

	std::fstream fs((ctx.output_path + "/" + filename + ".xdmf").c_str(),
			std::fstream::out);
	fs << xmdf_file;
	fs.close();
}

} // namespace io
} // namespace simpla

/*
 * wrte_xdmf.cpp
 *
 *  Created on: 2012-9-30
 *      Author: salmon
 */

#include "write_xdmf.h"

#include <string>
#include <fstream>
#include <sstream>

#include <H5Cpp.h>
#include <hdf5_hl.h>

#include "io/write_hdf5.h"

namespace simpla
{
namespace io
{
using namespace fetl;

inline std::string ToString(std::string const &v)
{
	return (v);
}
template<typename T> inline std::string ToString(T const &v)
{
	std::stringstream stream;
	stream << v;
	return (stream.str());
}
template<int N, typename T> inline std::string ToString(nTuple<N, T> const &v)
{
	std::stringstream stream;
	for (int i = 0; i < N; ++i)
	{
		stream << v[i] << " ";
	}
	return (stream.str());
}

void WriteXDMF(std::list<Object::Holder> const & objs,
		std::string const & dir_path, UniformRectGrid const & grid,
		TR1::function<Real()> counter)
{
	mkdir(dir_path.c_str(), 0777);

	std::stringstream st;
	st << std::setw(8) << std::setfill('0') << counter();
	std::string filename = st.str();

	std::fstream fs((dir_path + "/" + filename + ".xmf").c_str(),
			std::fstream::out);

	H5::Group grp = H5::H5File(dir_path + "/" + filename + ".h5", H5F_ACC_TRUNC) //
	.openGroup("/");

	size_t count = counter();

	Real time = count * grid.dt;
	fs << "<?xml version='1.0' ?>\n"

	<< "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>\n"

	<< "<Xdmf Version='2.0'>\n"

	<< "<Domain>\n"

	if (grid.nd == 3) //3D
	{

		fs << "<Grid Name='GRID" << count << "' GridType='Uniform' >\n"

		<< "  <Topology TopologyType='3DCoRectMesh'  "

		<< " Dimensions='" << ToString(grid.dims) << "'></Topology>\n"

		<< "  <Geometry Type='Origin_DxDyDz'>\n"

		<< "    <DataItem Format='XML' Dimensions='3'>"

		<< ToString(grid.xmin) << "</DataItem>\n"

		<< "    <DataItem Format='XML' Dimensions='3'>"

		<< ToString(grid.dx) << "</DataItem>\n"

		<< "  </Geometry>\n";

	}
	else if (grid.nd == 2) //2D
	{
		nTuple<TWO, size_t> dims =
		{ grid.dims[0], grid.dims[1] };
		nTuple<TWO, Real> xmin =
		{ grid.xmin[0], grid.xmin[1] };
		nTuple<TWO, Real> dx =
		{ grid.dx[0], grid.dx[1] };

		fs << "<Grid Name='GRID" << count << "' GridType='Uniform' >\n"

		<< "  <Topology TopologyType='2DCoRectMesh'  "

		<< " Dimensions='" << ToString(dims) << "'></Topology>\n"

		<< "  <Geometry Type='Origin_DxDyDz'>\n"

		<< "    <DataItem Format='XML' Dimensions='2'>"

		<< ToString(xmin) << "</DataItem>\n"

		<< "    <DataItem Format='XML' Dimensions='2'>"

		<< ToString(dx) << "</DataItem>\n"

		<< "  </Geometry>\n";
	}

	fs << "  <Time  Value='" << time << "' />\n";

	std::string attr_template = ""

	for (std::list<Object::Holder>::const_iterator it = objs.begin();
			it != objs.end(); ++it)
	{

		std::string name = (*it)->properties.get<std::string>("name");

		int ndim = (*it)->get_dimensions();

		size_t dims[ndim];

		(*it)->get_dimensions(dims);

		std::string dims_str = "";

		for (int i = 0; i < ndim; ++i)
		{
			dims_str += ToString(dims[i]) + " ";

		}

		std::string attr = attr_template;

		fs << "  <Attribute Name='" << name
				<< "'  AttributeType='Vector' Center='Node' >\n"

				<< "    <DataItem  NumberType='Float' Precision='8' Format='HDF' Dimensions='"

				<< dims_str << "' >\n"

				<< count << ".h5:/" << name

				<< "    </DataItem>\n"

				<< "  </Attribute>\n";

		WriteHDF5(grp, name, (*it));
	}

	fs << "</Grid>\n"
			"</Domain>\n"
			"</Xdmf>\n";
	fs.close();

}

} // namespace io
} // namespace simpla

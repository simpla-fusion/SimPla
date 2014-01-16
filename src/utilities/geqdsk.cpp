/*
 * read_geqdsk.cpp
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#include "geqdsk.h"

#include <XdmfArray.h>
#include <XdmfAttribute.h>
#include <XdmfDataDesc.h>
#include <XdmfDataItem.h>
#include <XdmfDOM.h>
#include <XdmfDomain.h>
#include <XdmfElement.h>
#include <XdmfGeometry.h>
#include <XdmfGrid.h>
#include <XdmfLightData.h>
#include <XdmfObject.h>
#include <XdmfRoot.h>
#include <XdmfTopology.h>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../io/xdmf_io.h"
#include "../numeric/interpolation.h"
#include "pretty_stream.h"
class XdmfArray;

namespace simpla
{

void GEqdsk::Read(std::string const &fname)
{
	std::ifstream inFileStream_;

	if (fname != "")
		inFileStream_.open(fname);
	Real rdim; // Horizontal dimension in meter of computational box
	Real zdim; // Vertical dimension in meter of computational box
	Real rleft; // Minimum R in meter of rectangular computational box
	Real zmid; // Z of center of computational box in meter
	Real simag; // Poloidal flux at magnetic axis in Weber / rad
	Real sibry; // Poloidal flux at the plasma boundary in Weber / rad
	size_t idum;
	Real xdum;

	inFileStream_.get(desc, 48);

	inFileStream_ >> std::setw(4) >> idum >> dims_[0] >> dims_[1];

	inFileStream_ >> std::setw(16)

	>> rdim >> zdim >> rcentr >> rleft >> zmid

	>> rmaxis >> zmaxis >> simag >> sibry >> bcentr

	>> current >> simag >> xdum >> rmaxis >> xdum

	>> zmaxis >> xdum >> sibry >> xdum >> xdum;

	rzmin_[0] = rleft;
	rzmax_[0] = rleft + rdim;

	rzmin_[1] = zmid - zdim / 2;
	rzmax_[1] = zmid + zdim / 2;

	inter2d_type(dims_, rzmin_, rzmax_).swap(psirz_);

#define INPUT_VALUE(_NAME_)                                                            \
	for (size_t s = 0; s < dims_[0]; ++s)                                              \
	{                                                                                  \
		value_type y;                                                                  \
		inFileStream_ >> std::setw(16) >> y;                                           \
		_NAME_.data().emplace(                                                         \
	     std::make_pair(static_cast<value_type>(s)                                     \
	          /static_cast<value_type>(dims_[0]-1), y));                               \
	}                                                                                  \

	INPUT_VALUE(fpol_);
	INPUT_VALUE(pres_);
	INPUT_VALUE(ffprim_);
	INPUT_VALUE(pprim_);

	for (size_t s = 0, s_end = dims_[0] * dims_[1]; s < s_end; ++s)
	{
		value_type v;
		inFileStream_ >> std::setw(16) >> v;
		psirz_.data().push_back((v - simag) / (sibry - simag)); // Normalize Poloidal flux
	}

	INPUT_VALUE(qpsi_);

#undef INPUT_VALUE

	unsigned int nbbbs, limitr;
	inFileStream_ >> std::setw(5) >> nbbbs >> limitr;
	CHECK(nbbbs);
	rzbbb_.resize(nbbbs);
	rzlim_.resize(limitr);

	inFileStream_ >> std::setw(16) >> rzbbb_;
	inFileStream_ >> std::setw(16) >> rzlim_;


}


std::ostream & GEqdsk::Print(std::ostream & os)
{
	std::cout << "--" << desc << std::endl;

//	std::cout << "nw" << "\t= " << nw
//			<< "\t--  Number of horizontal R grid  points" << std::endl;
//
//	std::cout << "nh" << "\t= " << nh << "\t-- Number of vertical Z grid points"
//			<< std::endl;
//
//	std::cout << "rdim" << "\t= " << rdim
//			<< "\t-- Horizontal dimension in meter of computational box                   "
//			<< std::endl;
//
//	std::cout << "zdim" << "\t= " << zdim
//			<< "\t-- Vertical dimension in meter of computational box                   "
//			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
	        << "\t--                                                                    " << std::endl;

//	std::cout << "rleft" << "\t= " << rleft
//			<< "\t-- Minimum R in meter of rectangular computational box                "
//			<< std::endl;
//
//	std::cout << "zmid" << "\t= " << zmid
//			<< "\t-- Z of center of computational box in meter                          "
//			<< std::endl;

	std::cout << "rmaxis" << "\t= " << rmaxis
	        << "\t-- R of magnetic axis in meter                                        " << std::endl;

	std::cout << "rmaxis" << "\t= " << zmaxis
	        << "\t-- Z of magnetic axis in meter                                        " << std::endl;

//	std::cout << "simag" << "\t= " << simag
//			<< "\t-- poloidal flus ax magnetic axis in Weber / rad                      "
//			<< std::endl;
//
//	std::cout << "sibry" << "\t= " << sibry
//			<< "\t-- Poloidal flux at the plasma boundary in Weber / rad                "
//			<< std::endl;

	std::cout << "rcentr" << "\t= " << rcentr
	        << "\t-- R in meter of  vacuum toroidal magnetic field BCENTR               " << std::endl;

	std::cout << "bcentr" << "\t= " << bcentr
	        << "\t-- Vacuum toroidal magnetic field in Tesla at RCENTR                  " << std::endl;

	std::cout << "current" << "\t= " << current
	        << "\t-- Plasma current in Ampere                                          " << std::endl;

//	std::cout << "fpol" << "\t= "
//			<< "\t-- Poloidal current function in m-T<< $F=RB_T$ on flux grid           "
//			<< std::endl << fpol_.data() << std::endl;
//
//	std::cout << "pres" << "\t= "
//			<< "\t-- Plasma pressure in $nt/m^2$ on uniform flux grid                   "
//			<< std::endl << pres_.data() << std::endl;
//
//	std::cout << "ffprim" << "\t= "
//			<< "\t-- $FF^\\prime(\\psi)$ in $(mT)^2/(Weber/rad)$ on uniform flux grid     "
//			<< std::endl << ffprim_.data() << std::endl;
//
//	std::cout << "pprim" << "\t= "
//			<< "\t-- $P^\\prime(\\psi)$ in $(nt/m^2)/(Weber/rad)$ on uniform flux grid    "
//			<< std::endl << pprim_.data() << std::endl;
//
//	std::cout << "psizr"
//			<< "\t-- Poloidal flus in Webber/rad on the rectangular grid points         "
//			<< std::endl << psirz_.data() << std::endl;
//
//	std::cout << "qpsi" << "\t= "
//			<< "\t-- q values on uniform flux grid from axis to boundary                "
//			<< std::endl << qpsi_.data() << std::endl;
//
//	std::cout << "nbbbs" << "\t= " << nbbbs
//			<< "\t-- Number of boundary points                                          "
//			<< std::endl;
//
//	std::cout << "limitr" << "\t= " << limitr
//			<< "\t-- Number of limiter points                                           "
//			<< std::endl;
//
//	std::cout << "rzbbbs" << "\t= "
//			<< "\t-- R of boundary points in meter                                      "
//			<< std::endl << rzbbb_ << std::endl;
//
//	std::cout << "rzlim" << "\t= "
//			<< "\t-- R of surrounding limiter contour in meter                          "
//			<< std::endl << rzlim_ << std::endl;

	return os;
}

void GEqdsk::Write(std::string const &fname, int flag)
{

	if (flag == XDMF)
	{

		XdmfDOM dom;
		XdmfRoot root;
		root.SetDOM(&dom);
		root.SetVersion(2.0);
		root.Build();

		XdmfDomain domain;
		root.Insert(&domain);

		{
			XdmfGrid grid;
			domain.Insert(&grid);

			grid.SetName("G-Eqdsk");
			grid.SetGridType(XDMF_GRID_UNIFORM);
			grid.GetTopology()->SetTopologyTypeFromString("2DCoRectMesh");

			XdmfInt64 dims[2] = { static_cast<XdmfInt64>(dims_[1]), static_cast<XdmfInt64>(dims_[0]) };
			grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);

			grid.GetGeometry()->SetGeometryTypeFromString("Origin_DxDy");
			grid.GetGeometry()->SetOrigin(rzmin_[1], rzmin_[0], 0);
			grid.GetGeometry()->SetDxDyDz((rzmax_[1] - rzmin_[1]) / static_cast<Real>(dims_[1] - 1),
			        (rzmax_[0] - rzmin_[0]) / static_cast<Real>(dims_[0] - 1), 0);

			XdmfAttribute myAttribute;
			grid.Insert(&myAttribute);

			myAttribute.SetName("Psi");
			myAttribute.SetAttributeTypeFromString("Scalar");
			myAttribute.SetAttributeCenterFromString("Node");

			XdmfDataItem data;
			myAttribute.Insert(&data);

			InsertDataItem(&data, 2, dims, &(psirz_.data()[0]), fname + ".h5:/Psi");
			grid.Build();
		}
		{
			XdmfGrid grid;
			domain.Insert(&grid);
			grid.SetName("Boundary");
			grid.SetGridType(XDMF_GRID_UNIFORM);
			grid.GetTopology()->SetTopologyTypeFromString("POLYLINE");

			XdmfInt64 dims[2] = { static_cast<XdmfInt64>(rzbbb_.size()), 2 };
			grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);
			grid.GetTopology()->Set("NodesPerElement", "2");
			grid.GetTopology()->SetNumberOfElements(rzbbb_.size());

			XdmfDataItem * data = new XdmfDataItem;

			grid.GetTopology()->Insert(data);

			InsertDataItemWithFun(data, 2, dims, [&](XdmfInt64 *d)->unsigned int
			{
				return d[1]==0?d[0]:(d[0]+1)%dims[0];
			},

			fname + ".h5:/Boundary/Topology");

			grid.GetGeometry()->SetGeometryTypeFromString("XYZ");

			data = new XdmfDataItem;
			data->SetHeavyDataSetName((fname + ".h5:/Boundary/Points").c_str());

			grid.GetGeometry()->Insert(data);

			XdmfArray *points = grid.GetGeometry()->GetPoints();

			dims[1] = 3;
			points->SetShape(2, dims);

			XdmfInt64 s = 0;
			for (auto const &v : rzbbb_)
			{
				points->SetValue(s * 3, 0);
				points->SetValue(s * 3 + 1, v[0]);
				points->SetValue(s * 3 + 2, v[1]);

				++s;
			}

			grid.Build();
		}
		{
			XdmfGrid grid;
			domain.Insert(&grid);
			grid.SetName("Limter");
			grid.SetGridType(XDMF_GRID_UNIFORM);
			grid.GetTopology()->SetTopologyTypeFromString("POLYLINE");

			XdmfInt64 dims[2] = { static_cast<XdmfInt64>(rzlim_.size()), 2 };
			grid.GetTopology()->GetShapeDesc()->SetShape(2, dims);
			grid.GetTopology()->Set("NodesPerElement", "2");
			grid.GetTopology()->SetNumberOfElements(rzlim_.size());

			XdmfDataItem * data = new XdmfDataItem;

			grid.GetTopology()->Insert(data);

			InsertDataItemWithFun(data, 2, dims, [&](XdmfInt64 *d)->unsigned int
			{
				return d[1]==0?d[0]:(d[0]+1)%dims[0];
			},

			fname + ".h5:/Limter/Topology");

			grid.GetGeometry()->SetGeometryTypeFromString("XYZ");

			data = new XdmfDataItem;
			data->SetHeavyDataSetName((fname + ".h5:/Limter/Points").c_str());

			grid.GetGeometry()->Insert(data);

			XdmfArray *points = grid.GetGeometry()->GetPoints();

			dims[1] = 3;
			points->SetShape(2, dims);

			XdmfInt64 s = 0;
			for (auto const &v : rzlim_)
			{
				points->SetValue(s * 3, 0);
				points->SetValue(s * 3 + 1, v[0]);
				points->SetValue(s * 3 + 2, v[1]);

				++s;
			}

			grid.Build();
		}

//		root.Build();
		std::ofstream ss(fname + ".xmf");
		ss << dom.Serialize() << endl;

	}

//	if (flag == XDMF)
//	{
//		std::ofstream ss(fname + ".xmf");
//
//		std::string h5name = fname + ".h5";
//
//		HDF5::H5OutStream h5s(h5name);
//
//		ss
//				<< "<?xml version='1.0' ?>                                              \n"
//				<< "<!DOCTYPE Xdmf SYSTEM 'Xdmf.dtd' []>                                \n"
//				<< "<Xdmf Version='2.0'>                                                \n"
//				<< "<Domain>                                                         \n";
//
//		// psi
//		ss
//
//		<< "<Grid Name='G_EQDSK' GridType='Uniform' >                     \n"
//
//		<< "  <Topology TopologyType='2DCoRectMesh'  Dimensions='" << dims_
//				<< "'>" << "   </Topology>\n"
//
//				<< "  <Geometry Type='Origin_DxDy'>                                 \n"
//				<< "     <DataItem Format='XML' Dimensions='2'> " << rzmin_[1]
//				<< " " << rzmin_[0] << "</DataItem>\n"
//				<< "     <DataItem Format='XML' Dimensions='2'>"
//				<< (rzmax_[1] - rzmin_[1]) / static_cast<Real>(dims_[1] - 1)
//				<< " "
//				<< (rzmax_[0] - rzmin_[0]) / static_cast<Real>(dims_[0] - 1)
//				<< "     </DataItem>\n"
//				<< "  </Geometry>                                                   \n"
//
//				<< "<!-- ADD_ATTRIBUTE_START -->                                        \n"
//
//				<< "  <Attribute Name='psi'  AttributeType='Scalar' Center='Node' >     \n"
//				<< "    <DataItem Format=\"HDF\"  NumberType='Float'"
//				<< " Precision='8'  Dimensions='" << dims_ << "'>          \n"
//				<< h5name << ":/psi"
//				<< "    </DataItem>                                                     \n"
//				<< "  </Attribute>                                                      \n"
//
//				<< "<!-- ADD_ATTRIBUTE_DONE -->                                         \n"
//
//				<< "</Grid>                                                             \n";
//
//		// boundary
//		ss
//				<< "<Grid Name=\"Boundary\" GridType=\"Uniform\">                       \n"
//				<< "   <Topology TopologyType=\"Polyline\" NodesPerElement='2'	"
//				<< "  NumberOfElements = '" << rzbbb_.size() << "' >  \n"
//				<< "       <DataItem Format=\"XML\"  Dimensions=\""
//				<< rzbbb_.size() << " 2\"   NumberType=\"UInt\">\n";
//		for (size_t s = 0, s_end = rzbbb_.size() - 1; s < s_end; ++s)
//		{
//			ss << s << " " << s + 1 << " ";
//		}
//
//		ss << rzbbb_.size() - 1 << " " << 0 << std::endl;
//
//		ss << "  </DataItem> </Topology>                 \n"
//
//				<< "   <Geometry Type=\"XYZ\">                                           \n"
//				<< "       <DataItem Format=\"XML\"  NumberType=\"Float\"  Dimensions=\""
//				<< rzbbb_.size() << " 3\"" << "> \n";
//		for (auto & v : rzbbb_)
//		{
//
//			ss << " 0 " << v[0] << " " << v[1] << std::endl;
//		}
//
//		ss
//				<< "      </DataItem>                                                   \n"
//				<< "   </Geometry>                                                      \n"
//
//				<< "</Grid>                                                             \n";
//
//		// limiter
//		ss
//				<< "<Grid Name=\"limiter\" GridType=\"Uniform\">                        \n"
//				<< "   <Topology TopologyType=\"Polyline\"  NodesPerElement='2'	"
//				<< "  NumberOfElements = '" << rzlim_.size() << "' >  \n"
//				<< "     <DataItem	Format = \"XML\"  Dimensions=\""
//				<< rzlim_.size() << " 2\"   NumberType=\"UInt\"> \n";
//
//		for (size_t s = 0, s_end = rzlim_.size() - 1; s < s_end; ++s)
//		{
//			ss << s << " " << s + 1 << " ";
//		}
//		ss << rzlim_.size() - 1 << " " << 0 << " " << std::endl;
//
//		ss << "  </DataItem> </Topology>                 \n"
//
//				<< "<Geometry Type=\"XYZ\">                                           \n"
//				<< "  <DataItem Format=\"XML\"  NumberType=\"Float\" Dimensions=\""
//				<< rzlim_.size() << " 3\" > \n";
//
//		for (auto & v : rzlim_)
//		{
//			ss << " 0 " << v[0] << " " << v[1] << std::endl;
//		}
//
//		ss
//				<< "  </DataItem>                                                  \n"
//				<< "</Geometry>                                                      \n"
//				<< "</Grid>                                                             \n"
//
//				;
//
//		ss
//		<< "</Domain>" << std::endl << "</Xdmf>" << std::endl;
//
//		h5s << HDF5::OpenDataSet("psi") << HDF5::SetDims(dims_) << psirz_.data()
//				<< HDF5::CloseDataSet();
//		//h5s << HDF5::OpenDataSet("rzbbb") << rzbbb_ << HDF5::CloseDataSet();
//		//h5s << HDF5::OpenDataSet("rzlim") << rzlim_ << HDF5::CloseDataSet();
//	}

}
}  // namespace simpla


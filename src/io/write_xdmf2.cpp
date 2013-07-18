/*
 * write_xdmf2.cpp
 *
 *  Created on: 2012-10-27
 *      Author: salmon
 */
#include "write_xdmf2.h"
#include "grid/uniform_rect.h"
#include <Xdmf.h>
#include <H5Cpp.h>
#include <hdf5_hl.h>
#include <sys/stat.h>
#include "fetl/fetl.h"
namespace simpla
{

namespace io
{

template<>
WriteXDMF2<UniformRectGrid>::~WriteXDMF2()
{
}
template<>
void WriteXDMF2<UniformRectGrid>::Initialize()
{
	mkdir(path_.c_str(), 0777);

	LOG << "Create module WriteXDMF";
}

template<>
void WriteXDMF2<UniformRectGrid>::Eval()
{
	typedef UniformRectGrid Grid;

	LOG << "Run module WriteXDMF2";
	XdmfDOM dom;
	XdmfDomain domain;
	XdmfRoot root;
	root.SetDOM(&dom);
	root.SetVersion(2.2);
	root.Build();
	root.Insert(&domain);
	XdmfGrid xgrid;

	domain.Insert(&xgrid);

	int ndims = 0;
	XdmfFloat64 xmin[3], dx[3];
	XdmfInt64 dims[3];
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
	XdmfTopology * topology = xgrid.GetTopology();

	if (ndims = 2)
	{
		topology->SetTopologyType(XDMF_2DCORECTMESH);
	}
	else if (ndims = 3)
	{
		topology->SetTopologyType(XDMF_3DCORECTMESH);
	}

	topology->GetShapeDesc()->SetShape(ndims, dims);

	XdmfGeometry * geo = xgrid.GetGeometry();
	geo->SetOrigin(xmin);
	geo->SetDxDyDz(dx);

	xgrid.GetTime()->SetValue(ctx.Timer());

	std::string filename;

	{
		std::ostringstream st;
		st << std::setw(8) << std::setfill('0') << ctx.Counter();
		filename = st.str();
	}

	H5::Group grp = H5::H5File(path_ + "/" + filename + ".h5", H5F_ACC_TRUNC) //
	.openGroup("/");

	hsize_t mdims[MAX_XDMF_NDIMS];

	for (std::list<std::string>::const_iterator it = obj_list_.begin();
			it != obj_list_.end(); ++it)
	{
		std::map<std::string, TR1::shared_ptr<NdArray> >::const_iterator oit =
				ctx.objects.find(*it);

		if (oit != ctx.objects.end())
		{

			XdmfAttribute attr;
			attr.SetName(it->c_str());
			attr.SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);

			xgrid.Insert(&attr);
			root.Build();

			NdArray & obj = *oit->second;

			H5::DataType mdatatype(
					H5LTtext_to_dtype(obj.get_element_type_desc().c_str(),
							H5LT_DDL));

			int ndstart = 0;

			if (obj.CheckType(typeid(Field<Grid, IZeroForm, Real> )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
			}
			else if (obj.CheckType(typeid(Field<Grid, IZeroForm, Complex> )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
			}
			else if (obj.CheckType(typeid(Field<Grid, IOneForm, Real> ))
					|| obj.CheckType(typeid(Field<Grid, ITwoForm, Real> )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_VECTOR);
			}
			else if (obj.CheckType(typeid(Field<Grid, IOneForm, Complex> ))
					|| obj.CheckType(typeid(Field<Grid, ITwoForm, Complex> )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_VECTOR);
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Real> > )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_VECTOR);
				mdatatype = H5::PredType::NATIVE_DOUBLE;
				ndstart = 1;
				mdims[0] = THREE;
			}
			else if (obj.CheckType(
					typeid(Field<Grid, IZeroForm, nTuple<THREE, Complex> > )))
			{
				attr.SetAttributeType(XDMF_ATTRIBUTE_TYPE_VECTOR);
				mdatatype = H5LTtext_to_dtype(
						DataType<Complex>().desc().c_str(), H5LT_DDL);
				ndstart = 1;
				mdims[0] = THREE;
			}

			obj.get_dimensions(mdims + ndstart);

			int hnd = 0;
			hsize_t hmdims[MAX_XDMF_NDIMS];
			for (int i = 0; i < ndims + ndstart; ++i)
			{
				if (mdims[i] > 1)
				{
					hmdims[hnd] = mdims[i];
					++hnd;
				}
			}

			H5::DataSet dataset = grp.createDataSet((*it).c_str(), mdatatype,
					H5::DataSpace(hnd, hmdims));

			dataset.write(obj.get_data(), mdatatype);

		}
	}
	dom.Write((path_ + "/" + filename + ".xdmf").c_str());

}
}  // namespace io

}  // namespace simpla

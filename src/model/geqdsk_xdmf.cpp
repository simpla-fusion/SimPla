/*
 * geqdsk_xdmf.cpp
 *
 *  Created on: 2014年4月21日
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

#include "../io/xdmf_io.h"

class XdmfArray;
namespace simpla
{
void GEqdsk::Write(std::string const &fname, int flag)
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

		InsertDataItem(&data, 2, dims, &(psirz_[0]), fname + ".h5:/Psi");
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

		fname + ".h5:/Limter/Topology");

		grid.GetGeometry()->SetGeometryTypeFromString("XYZ");

		data = new XdmfDataItem;
		data->SetHeavyDataSetName((fname + ".h5:/Limter/Points").c_str());

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

//		root.Build();
	std::ofstream ss(fname + ".xmf");
	ss << dom.Serialize() << endl;

}
}  // namespace simpla

/*
 * test3.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <random>

#include "utilities/log.h"

#include "fetl/fetl.h"

#include "mesh/co_rect_mesh.h"

using namespace simpla;

DEFINE_FIELDS(CoRectMesh<>)
int main(int argc, char** argv)
{
	typedef VectorForm<0> FieldType;
	typedef Form<0> RScalarField;

	Mesh mesh;
	mesh.dt_ = 1.0;
	mesh.xmin_[0] = 0;
	mesh.xmin_[1] = 0;
	mesh.xmin_[2] = 0;
	mesh.xmax_[0] = 1.0;
	mesh.xmax_[1] = 1.0;
	mesh.xmax_[2] = 1.0;
	mesh.dims_[0] = 20;
	mesh.dims_[1] = 30;
	mesh.dims_[2] = 40;
	mesh.gw_[0] = 2;
	mesh.gw_[1] = 2;
	mesh.gw_[2] = 2;

	mesh.Update();

	FieldType f1(mesh), f2(mesh), f3(mesh);

	Real a, b, c;
	a = 1.0, b = -2.0, c = 3.0;

	typedef typename FieldType::value_type value_type;

	value_type va, vb;

	va = 2.0;
	vb = 3.0;

	std::fill(f1.begin(), f1.end(), va);
	std::fill(f2.begin(), f2.end(), vb);

	auto vv = f1.get(0, 0, 0, 0) / 2.0;
	f3 = -f2 + f2 / c - f1 * a;
}

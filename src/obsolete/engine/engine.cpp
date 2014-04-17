/*
 * engine.cpp
 *
 *  Created on: 2013年11月19日
 *      Author: salmon
 */

#include <engine/engine.h>
#include <engine/object.h>
#include <fetl/ntuple.h>
#include "mesh/uniform_rect.h"
#include <sstream>
#include <string>

#include "../../third_part/pugixml/src/pugixml.hpp"

namespace simpla
{

template<int N, typename T>
nTuple<N, T> nTupleFromString(std::string const & str)
{
	std::istringstream is(str);
	nTuple<N, T> tv;
	for (int i = 0; i < N && is; ++i)
	{
		is >> tv[i];
	}
	return tv;
}
template<int N, typename T>
std::string nTupleToString(nTuple<N, T> const & v)
{
	std::ostringstream os;

	for (int i = 0; i < N; ++i)
	{
		os << " " << v[i];
	}
	return os.str();
}

template<typename TM, typename TV>
Object _CreateField(TM const &mesh, pugi::xml_node const & node)
{
	auto center = node.attribute("Center").as_string();

	typedef TM mesh_type;
	Object res;
	if (center == "Node")
	{
		res = Object(new Field<Geometry<mesh_type, 0>, TV>(mesh));
	}
	else if (center == "Edge")
	{
		res = Object(new Field<Geometry<mesh_type, 1>, TV>(mesh));
	}
	else if (center == "Face")
	{
		res = Object(new Field<Geometry<mesh_type, 2>, TV>(mesh));
	}
	else if (center == "Cell")
	{
		res = Object(new Field<Geometry<mesh_type, 3>, TV>(mesh));
	}
}
template<typename TM>
std::pair<std::string, Object> CreateAttribute(TM const & mesh,
		pugi::xml_node const & node)
{
	typedef TM mesh_type;

	std::string name = node.attribute("Name").as_string();

	std::string type = node.attribute("AttributeType").as_string();

	Object res;

	if (type != "Particle")
	{

		if (type == "Scalar")
		{
			res = _CreateField<mesh_type, Real>(mesh, node);
		}
		else if (type == "Vector")
		{
			res = _CreateField<mesh_type, nTuple<3, Real>>(mesh, node);
		}
		else if (type == "Complex")
		{
			res = _CreateField<mesh_type, std::complex<Real>>(mesh, node);
		}
		else if (type == "ComplexVector")
		{
			res = _CreateField<mesh_type, nTuple<3, std::complex<Real>>>(mesh,
			node);
		}
	}
	else
	{
		auto type = node.attribute("Type").as_string();
	}

	return std::move(std::make_pair(name, res));
}

Object Engine::Evaluate()
{
	Object obj_grid;

	auto grid = doc_.select_single_node("/Xdmf/Domain/Grid").node();

	auto topology = doc_.child("Topology");

	auto geometry = doc_.child("Geometry");

	if (topology.attribute("Type").as_string() == "3DCoRectMesh")
	{
		CoRectMesh * mesh = new CoRectMesh();

		mesh->SetTopology(
				nTupleFromString<3, size_t>(
						topology.attribute("Dimensions").value()),
				nTupleFromString<3, size_t>(
						topology.attribute("GhostWidth").value()));
		mesh->SetGeometry(
				nTupleFromString<3, Real>(
						geometry.select_single_node("DataItem[0]").node().value()),
				nTupleFromString<3, Real>(
						geometry.select_single_node("DataItem[0]").node().value()));

		mesh->Update();

		obj_grid = Object(mesh);

		std::map<std::string, Real> constants_;

		for (auto const &p : grid.select_nodes(
				"Information[@Name='GlobalVaraible']/Variable"))
		{
			constants_[p.node().attribute("Name").as_string()] =
					p.node().attribute("Value").as_double();
		}

		for (auto const &p : grid.select_nodes("Attribute"))
		{
			obj_grid.GetChildren().insert(CreateAttribute(mesh, p.node()));
		}

	}

}

return obj_grid;
}

}  // namespace simpla


/*
 * field_convert.h
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#ifndef FIELD_CONVERT_H_
#define FIELD_CONVERT_H_

#include <type_traits>

namespace simpla
{
template<typename TS> class CoRectMesh;
template<typename, typename > class Field;
template<typename, int> class Geometry;

template<typename TS, typename T>
void MapTo(Field<Geometry<CoRectMesh<TS>, EDGE>, T> const & l,
		Field<Geometry<CoRectMesh<TS>, VERTEX>, nTuple<3, T>> * r)
{
	r->Init();

	typedef CoRectMesh<TS> mesh_type;

	mesh_type const &mesh = r->mesh;

	auto const &dims = mesh.GetDimension();

	typedef typename mesh_type::index_type index_type;

	mesh.ParallelTraversal(0,

	[&](int m, index_type const &x,index_type const &y,index_type const &z)
	{
		auto &v =r->get(0,x,y,z);
		for(int i=0;i<3;++i)
		{
			v[i]=(mesh.get(l,i,x,y,z)+mesh.get(l,i,mesh.Shift(mesh.DES(i),x,y,z)))*0.5;
		}
	});

}

template<typename TS, typename T>
void MapTo(Field<Geometry<CoRectMesh<TS>, VERTEX>, nTuple<3, T>> const & l,
		Field<Geometry<CoRectMesh<TS>, EDGE>, T> * r)
{
	r->Init();

	typedef CoRectMesh<TS> mesh_type;

	mesh_type const &mesh = l.mesh;

	auto const &dims = mesh.GetDimension();

	typedef typename mesh_type::index_type index_type;

	mesh.ParallelTraversal(0,

	[&](int m, index_type const &x,index_type const &y,index_type const &z)
	{
		for(int i=0;i<3;++i)
		{
			r->get(i,x,y,z)=(mesh.get(l,0,x,y,z)[i]+mesh.get(l,0,mesh.Shift(mesh.INC(i),x,y,z))[i])*0.5;
		}
	}

	);

}

}  // namespace simpla

#endif /* FIELD_CONVERT_H_ */

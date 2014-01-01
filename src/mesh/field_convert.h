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

	typedef typename mesh_type::index_type index_type;
	r->mesh.ParallelTraversal(0,

	[&](int m, index_type const &x,index_type const &y,index_type const &z)
	{
		auto &v =r->get(0,x,y,z);
		v[0]=(mesh.get(l,0,x,y,z)+mesh.get(l,0,mesh.Shift(mesh.DES(0),x,y,z)))*0.5;
		v[1]=(mesh.get(l,1,x,y,z)+mesh.get(l,1,mesh.Shift(mesh.DES(1),x,y,z)))*0.5;
		v[2]=(mesh.get(l,2,x,y,z)+mesh.get(l,2,mesh.Shift(mesh.DES(2),x,y,z)))*0.5;

	});

}

template<typename TS, typename T>
void MapTo(Field<Geometry<CoRectMesh<TS>, VERTEX>, nTuple<3, T>> const & l,
		Field<Geometry<CoRectMesh<TS>, EDGE>, T> * r)
{
	r->Init();

	typedef CoRectMesh<TS> mesh_type;

	mesh_type const &mesh = l.mesh;
	typedef typename mesh_type::index_type index_type;

	mesh.ParallelTraversal(0,

	[&](int m, index_type const &x,index_type const &y,index_type const &z)
	{
		auto &v =r->get(0,x,y,z);
		r->get(0,x,y,z)=(mesh.get(l,0,x,y,z)[0]+mesh.get(l,0,mesh.Shift(mesh.INC(0),x,y,z))[0])*0.5;
		r->get(1,x,y,z)=(mesh.get(l,0,x,y,z)[1]+mesh.get(l,0,mesh.Shift(mesh.INC(1),x,y,z))[1])*0.5;
		r->get(2,x,y,z)=(mesh.get(l,0,x,y,z)[2]+mesh.get(l,0,mesh.Shift(mesh.INC(2),x,y,z))[2])*0.5;
	}

	);

}

}  // namespace simpla

#endif /* FIELD_CONVERT_H_ */

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

template<typename TS, int IL, typename T>
typename std::enable_if<IL == 1 || IL == 2, void>::type //
MapTo(Field<Geometry<CoRectMesh<TS>, IL>, T> const & l, Field<Geometry<CoRectMesh<TS>, 0>, nTuple<3, T>> * r)
{
	if (r->size() <= 0)
		r->Init();
	typedef CoRectMesh<TS> mesh_type;
	mesh_type const &mesh = r->mesh;

	mesh.TraversalIndex(0, [&](typename mesh_type::index_type const &s)
	{
		auto &v =r->get(0,s);
		v[0]=mesh.mapto(Int2Type<0>(),l,0,s);
		v[1]=mesh.mapto(Int2Type<0>(),l,1,s);
		v[2]=mesh.mapto(Int2Type<0>(),l,2,s);

	}, mesh_type::DO_PARALLEL);

}

template<typename TS, int IR, typename T>
typename std::enable_if<IR == 1 || IR == 2, void>::type //
MapTo(Field<Geometry<CoRectMesh<TS>, 0>, nTuple<3, T>> const & l, Field<Geometry<CoRectMesh<TS>, IR>, T> * r)
{
	if (r->size() <= 0)
		r->Init();

	typedef CoRectMesh<TS> mesh_type;

	mesh_type const &mesh = l.mesh;

	mesh.TraversalIndex(0, [&](typename mesh_type::index_type const &s)
	{
		r->get(0,s)=mesh.mapto(Int2Type<IR>(),l,0,s)[0];
		r->get(1,s)=mesh.mapto(Int2Type<IR>(),l,1,s)[1];
		r->get(2,s)=mesh.mapto(Int2Type<IR>(),l,2,s)[2];

	}, mesh_type::DO_PARALLEL);

}

}  // namespace simpla

#endif /* FIELD_CONVERT_H_ */

/*
 * rect_mesh.h
 *
 *  Created on: 2014年2月26日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

#include "../fetl/primitives.h"
#include "../utilities/type_utilites.h"
#include "octree_forest.h"

namespace simpla
{

class RectMesh: public OcForest
{

	//! VERTEX -> EDGE
	template<typename TF>
	auto Grad(TF const & f,
			index_type s) const
					DECL_RET_TYPE(
							( (f[s + s & (_MA >> (s.H + 1))] - f[s - s & (_MA >> (s.H + 1))]) * static_cast<double>(1UL << s.H) )
					)

	//! VERTEX -> EDGE
	template<typename TF>
	auto Diverge(TF const & f, index_type s, Real const a[3]) const
	DECL_RET_TYPE(
			((
							(f[s + (_MI >> (s.H + 1))] - f[s - (_MI >> (s.H + 1))]) *a[0]+
							(f[s + (_MJ >> (s.H + 1))] - f[s - (_MJ >> (s.H + 1))]) *a[1]+
							(f[s + (_MK >> (s.H + 1))] - f[s - (_MK >> (s.H + 1))]) *a[2]
					)* static_cast<double>(1UL << s.H) )
	)

	//! Curl(Field<Edge>) Edge=>FACE
	template<typename TF>
	auto Curl(TF const & f, Int2Type<EDGE>,
			index_type s, //! this is FACE index
			Real const a[3]) const
					DECL_RET_TYPE(
							((
											(f[ s + (_R( _I(s))>> (s.H +1) ) ] - f[s - (_R( _I(s))>> (s.H +1) )]) *a[ _N(_R( _I(s))) ]-
											(f[ s + (_RR( _I(s))>> (s.H +1) ) ] - f[s - (_RR( _I(s))>> (s.H +1) )]) *a[ _N(_RR( _I(s))) ]
									)* static_cast<double>(1UL << s.H) )
					)

	//! Curl(Field<FACE>) FACE=>EDGE
	template<typename TF>
	auto Curl(TF const & f, Int2Type<FACE>,
			index_type s, //! this is edge index
			Real const a[3]) const
					DECL_RET_TYPE(
							((
											(f[ s + (_R( s)>> (s.H +1) ) ] - f[s - (_R( s)>> (s.H +1) )]) *a[ _N(_R( s))]-
											(f[ s + (_RR( s)>> (s.H +1) )] - f[s - (_RR( s)>> (s.H +1) )]) *a[ _N(_RR( s))]

									)* static_cast<double>(1UL << s.H) )
					)

	//***************************************************************************************************
	//* Container/Field operation
	//* Field vs. Mesh
	//***************************************************************************************************

	template<typename TL, typename TR> void AssignContainer(int IFORM, TL * lhs, TR const &rhs) const
	{
		ParallelTraversal(IFORM, [&]( index_type s)
		{	get(lhs,s)=get(rhs,s);});

	}

	template<typename T>
	inline typename std::enable_if<!is_field<T>::value, T>::type get(T const &l, index_type) const
	{
		return std::move(l);
	}

};

}  // namespace simpla

#endif /* RECT_MESH_H_ */

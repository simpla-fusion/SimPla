/*
 * uniform_rect_ops.h
 *
 *  Created on: 2013年10月28日
 *      Author: salmon
 */

#ifndef UNIFORM_RECT_OPS_H_
#define UNIFORM_RECT_OPS_H_

#include <fetl/expression.h>
#include <fetl/field.h>
#include <fetl/geometry.h>
#include <fetl/primitives.h>
#include <mesh/uniform_rect.h>
#include <type_traits>

namespace simpla
{

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

template<int N, typename TL> inline auto //
_Op(Int2Type<EXTRIORDERIVATIVE>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((f[s]*f.mesh->inv_dx[s%3]) )

template<typename TExpr> inline auto //
_Op(Int2Type<GRAD>, Field<Geometry<UniformRectMesh, 0>, TExpr> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[(s - s % 3) / 3 + f.mesh->strides[s % 3]] - f[(s - s % 3) / 3]) * f.mesh->inv_dx[s % 3])

template<typename TExpr> inline auto //
_Op(Int2Type<DIVERGE>, Field<Geometry<UniformRectMesh, 1>, TExpr> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(

						(f[s * 3 + 0] - f[s * 3 + 0 - 3 * f.mesh->strides[0]]) * f.mesh->inv_dx[0] +

						(f[s * 3 + 1] - f[s * 3 + 1 - 3 * f.mesh->strides[1]]) * f.mesh->inv_dx[1] +

						(f[s * 3 + 2] - f[s * 3 + 2 - 3 * f.mesh->strides[2]]) * f.mesh->inv_dx[2])

template<typename TL> inline auto //
_Op(Int2Type<CURL>, Field<Geometry<UniformRectMesh, 1>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[s - s %3 + (s + 2) % 3 + 3 * f.mesh->strides[(s + 1) % 3]]

								- f[s - s %3 + (s + 2) % 3]) * f.mesh->inv_dx[(s + 1) % 3]

						- (f[s - s %3 + (s + 1) % 3 + 3 * f.mesh->strides[(s + 2) % 3]]

								- f[s - s %3 + (s + 1) % 3]) * f.mesh->inv_dx[(s + 2) % 3])

template<typename TL> inline auto //
_Op(Int2Type<CURL>, Field<Geometry<UniformRectMesh, 2>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[s - s % 3 + (s + 2) % 3] - f[s - s % 3 + (s + 2) % 3 - 3 * f.mesh->strides[(s + 1) % 3]] )
						* f.mesh->inv_dx[(s + 1) % 3]

						-(f[s - s % 3 + (s + 1) % 3] - f[s - s % 3 + (s + 1) % 3 - 3 * f.mesh->strides[(s + 1) % 3]])
						* f.mesh->inv_dx[(s + 2) % 3])

//template<typename TL> inline auto //
//_Op(Int2Type<CURLPD1>, Field<Geometry<UniformRectMesh, 1>, TL> const & f,
//		typename UniformRectMesh::index_type s)
//				DECL_RET_TYPE(
//						(f.rhs_[s-s % 3 + 2 + 3 * f.mesh->strides[1]] - f.rhs_[s-s % 3 + 2]) * f.mesh->inv_dx[1])
//
//template<typename TL> inline auto //
//_Op(Int2Type<CURLPD2>, Field<Geometry<UniformRectMesh, 2>, TL> const & f,
//		typename UniformRectMesh::index_type s)
//				DECL_RET_TYPE(
//						(-f.rhs_[s-s % 3 + 1 + 3 * f.mesh->strides[2]] + f.rhs_[s-s % 3 + 1]) * f.mesh->inv_dx[2])

template<int IL, int IR, typename TL, typename TR> inline auto //
_Op(Int2Type<WEDGE>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(l.mesh->mapto(Int2Type<IL+IR>(),l,s)*r.mesh->mapto(Int2Type<IL+IR>(),r,s)))

template<int N, typename TL> inline auto //
_Op(Int2Type<HODGESTAR>, Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE((f.mesh->mapto(Int2Type<UniformRectMesh::NUM_OF_DIMS-N >(),f,s)))

template<int N, typename TL> inline auto //
_Op(Int2Type<NEGATE>, Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((-f[s]))

template<int IL, typename TL, typename TR> inline auto //
_Op(Int2Type<PLUS>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]+r[s]))

template<int IL, typename TL, typename TR> inline auto //
_Op(Int2Type<MINUS>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]-r[s]))

template<int IL, int IR, typename TL, typename TR> inline auto //
_Op(Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE( (l.mesh->mapto(Int2Type<IL+IR>(),l,s)*r.mesh->mapto(Int2Type<IL+IR>(),r,s)) )

template<int IL, typename TL, typename TR> inline std::enable_if<
		std::is_arithmetic<TR>::value,
		typename Field<Geometry<UniformRectMesh, IL>, TL>::value_type> //
_Op(Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, typename UniformRectMesh::index_type s)
{
	return l[s] * r;
}

template<int IR, typename TL, typename TR> inline std::enable_if<
		std::is_arithmetic<TL>::value,
		typename Field<Geometry<UniformRectMesh, IR>, TR>::value_type> //
_Op(Int2Type<MULTIPLIES>, TL const & l,
		Field<Geometry<UniformRectMesh, IR>, TR> const & r,
		typename UniformRectMesh::index_type s)
{
	return l * r[s];
}

template<int IL, typename TL, typename TR> inline auto //
_Op(Int2Type<DIVIDES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]/l.mesh->mapto(Int2Type<IL>(),r,s)))

//	template<typename TL, typename TR> inline auto //
//	Divides(Field<Geometry<ThisType, 0>, TL> const &l,
//			Field<Geometry<ThisType, 0>, TR> const &r, typename UniformRectMesh::index_type  s)
//			DECL_RET_TYPE((l[s]/r[s]))
//
//	template<int IL, typename TL, typename TR> inline auto //
//	Divides(Field<Geometry<ThisType, IL>, TL> const &l, TR r, typename UniformRectMesh::index_type  s)
//	DECL_RET_TYPE( (l[s]/r))
//
//
//	template<int IPD, typename TExpr> inline auto //	Field<Geometry<this_type, 2>,
//	OpCurlPD(Int2Type<IPD>, TExpr const & expr,
//			size_t  s) ->
//			typename std::enable_if<order_of_form<TExpr>::value==2, decltype(expr[0]) >::type
//	{
//		if (dims[IPD] == 1)
//		{
//			return (0);
//		}
//		size_t j0 = s % 3;
//
//		size_t idx2 = s - j0;
//
//		typename Field<Geometry<Mesh, 2>, TExpr>::Value res = 0.0;
////		if (1 == IPD)
////		{
////			res = (expr.rhs_[idx2 + 2]
////					- expr.rhs_[idx2 + 2 - 3 * strides[IPD]]) * f.mesh->inv_dx[IPD];
////
////		}
////		else if (2 == IPD)
////		{
////			res = (-expr.rhs_[idx2 + 1]
////					+ expr.rhs_[idx2 + 1 - 3 * strides[IPD]]) * f.mesh->inv_dx[IPD];
////		}
//
//		return (res);
//	}
}// namespace simpla

#endif /* UNIFORM_RECT_OPS_H_ */

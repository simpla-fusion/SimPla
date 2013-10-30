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

template<int N, typename TL> inline auto _OpEval(Int2Type<EXTRIORDERIVATIVE>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((f[s]*f.mesh->inv_dx_[s%3]))

template<typename TExpr> inline auto _OpEval(Int2Type<GRAD>,
		Field<Geometry<UniformRectMesh, 0>, TExpr> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[(s - s % 3) / 3 + f.mesh->strides_[s % 3]] - f[(s - s % 3) / 3]) * f.mesh->inv_dx_[s % 3])

template<typename TExpr> inline auto _OpEval(Int2Type<DIVERGE>,
		Field<Geometry<UniformRectMesh, 1>, TExpr> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(

						(f[s * 3 + 0] - f[s * 3 + 0 - 3 * f.mesh->strides_[0]]) * f.mesh->inv_dx_[0] +

						(f[s * 3 + 1] - f[s * 3 + 1 - 3 * f.mesh->strides_[1]]) * f.mesh->inv_dx_[1] +

						(f[s * 3 + 2] - f[s * 3 + 2 - 3 * f.mesh->strides_[2]]) * f.mesh->inv_dx_[2])

template<typename TL> inline auto _OpEval(Int2Type<CURL>,
		Field<Geometry<UniformRectMesh, 1>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[s - s %3 + (s + 2) % 3 + 3 * f.mesh->strides_[(s + 1) % 3]] - f[s - s %3 + (s + 2) % 3]) * f.mesh->inv_dx_[(s + 1) % 3]

						- (f[s - s %3 + (s + 1) % 3 + 3 * f.mesh->strides_[(s + 2) % 3]] - f[s - s %3 + (s + 1) % 3]) * f.mesh->inv_dx_[(s + 2) % 3]

				)

template<typename TL> inline auto _OpEval(Int2Type<CURL>,
		Field<Geometry<UniformRectMesh, 2>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(f[s - s % 3 + (s + 2) % 3] - f[s - s % 3 + (s + 2) % 3 - 3 * f.mesh->strides_[(s + 1) % 3]] ) * f.mesh->inv_dx_[(s + 1) % 3]

						-(f[s - s % 3 + (s + 1) % 3] - f[s - s % 3 + (s + 1) % 3 - 3 * f.mesh->strides_[(s + 2) % 3]]) * f.mesh->inv_dx_[(s + 2) % 3]

				)

//template<typename TL>  inline auto //
//_OpEval(<CURLPD1>, Field<Geometry<UniformRectMesh, 1>, TL> const & f,
//		typename UniformRectMesh::index_type s)
//				DECL_RET_TYPE(
//						(f.rhs_[s-s % 3 + 2 + 3 * f.mesh->strides_[1]] - f.rhs_[s-s % 3 + 2]) * f.mesh->inv_dx_[1])
//
//template<typename TL>  inline auto //
//_OpEval(<CURLPD2>, Field<Geometry<UniformRectMesh, 2>, TL> const & f,
//		typename UniformRectMesh::index_type s)
//				DECL_RET_TYPE(
//						(-f.rhs_[s-s % 3 + 1 + 3 * f.mesh->strides_[2]] + f.rhs_[s-s % 3 + 1]) * f.mesh->inv_dx_[2])

template<int IL, int IR, typename TL, typename TR> inline auto _OpEval(
		Int2Type<WEDGE>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE(
						(l.mesh->mapto(Int2Type<IL+IR>(),l,s)*r.mesh->mapto(Int2Type<IL+IR>(),r,s)))

template<int N, typename TL> inline auto _OpEval(Int2Type<HODGESTAR>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE((f.mesh->mapto(Int2Type<UniformRectMesh::NUM_OF_DIMS-N >(),f,s)))

template<int N, typename TL> inline auto _OpEval(Int2Type<NEGATE>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((-f[s]))

template<int IL, typename TL, typename TR> inline auto _OpEval(Int2Type<PLUS>,
		Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]+r[s]))

template<int IL, typename TL, typename TR> inline auto _OpEval(Int2Type<MINUS>,
		Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]-r[s]))

template<int IL, int IR, typename TL, typename TR> inline auto _OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		typename UniformRectMesh::index_type s)
				DECL_RET_TYPE( (l.mesh->mapto(Int2Type<IL+IR>(),l,s)*r.mesh->mapto(Int2Type<IL+IR>(),r,s)) )

template<int IL, typename TL, typename TR> inline auto _OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s] * r))

template<int IR, typename TL, typename TR> inline auto _OpEval(
		Int2Type<MULTIPLIES>, TL const & l,
		Field<Geometry<UniformRectMesh, IR>, TR> const & r,
		typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l * r[s]))

template<int IL, typename TL, typename TR> inline auto _OpEval(
		Int2Type<DIVIDES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, typename UniformRectMesh::index_type s)
		DECL_RET_TYPE((l[s]/l.mesh->mapto(Int2Type<IL>(),r,s)))

//template
//	Divides(Field<Geometry<ThisType, 0>, TL> const &l,
//			Field<Geometry<ThisType, 0>, TR> const &r, typename UniformRectMesh::index_type  s)
//			DECL_RET_TYPE((l[s]/r[s]))
//
//	template<int IL, typename TL, typename TR>  inline auto //
//	Divides(Field<Geometry<ThisType, IL>, TL> const &l, TR r, typename UniformRectMesh::index_type  s)
//	DECL_RET_TYPE( (l[s]/r))
//
//
//	template<int IPD, typename TExpr>  inline auto //	Field<Geometry<this_type, 2>,
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
////					- expr.rhs_[idx2 + 2 - 3 * strides_[IPD]]) * f.mesh->inv_dx_[IPD];
////
////		}
////		else if (2 == IPD)
////		{
////			res = (-expr.rhs_[idx2 + 1]
////					+ expr.rhs_[idx2 + 1 - 3 * strides_[IPD]]) * f.mesh->inv_dx_[IPD];
////		}
//
//		return (res);
//	}
}// namespace simpla

/**
 *
 *
 // Interpolation ----------------------------------------------------------

 template<typename TExpr>
 inline typename Field<Geometry<this_type, 0>, TExpr>::Value //
 Gather(Field<Geometry<this_type, 0>, TExpr> const &f, RVec3 x) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 return (f[s] * (1.0 - r[0]) + f[s + strides_[0]] * r[0]); //FIXME Only for 1-dim
 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 0>, TExpr> & f, RVec3 x,
 typename Field<Geometry<this_type, 0>, TExpr>::Value const v) const
 {
 typename Field<Geometry<this_type, 0>, TExpr>::Value res;
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f.Add(s, v * (1.0 - r[0]));
 f.Add(s + strides_[0], v * r[0]); //FIXME Only for 1-dim

 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 1>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (0.5 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (0.5 + r[0]));
 res[1] = (f[(s) * 3 + 1] * (0.5 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (0.5 + r[1]));
 res[2] = (f[(s) * 3 + 2] * (0.5 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (0.5 + r[2]));
 return res;
 }
 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 1>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 1>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx = r + 0.5;
 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
 f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
 f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
 }

 template<typename TExpr>
 inline nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> //
 Gather(Field<Geometry<this_type, 2>, TExpr> const &f, RVec3 x) const
 {
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> res;

 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 res[0] = (f[(s) * 3 + 0] * (1.0 - r[0])
 + f[(s - strides_[0]) * 3 + 0] * (r[0]));
 res[1] = (f[(s) * 3 + 1] * (1.0 - r[1])
 + f[(s - strides_[1]) * 3 + 1] * (r[1]));
 res[2] = (f[(s) * 3 + 2] * (1.0 - r[2])
 + f[(s - strides_[2]) * 3 + 2] * (r[2]));
 return res;

 }

 template<typename TExpr>
 inline void //
 Scatter(Field<Geometry<this_type, 2>, TExpr> & f, RVec3 x,
 nTuple<THREE, typename Field<Geometry<this_type, 2>, TExpr>::Value> const &v) const
 {
 IVec3 idx;
 Vec3 r;
 r = (x - xmin) * inv_dx_;
 idx[0] = static_cast<long>(r[0]);
 idx[1] = static_cast<long>(r[1]);
 idx[2] = static_cast<long>(r[2]);

 r -= idx;
 size_t s = idx[0] * strides_[0] + idx[1] * strides_[1]
 + idx[2] * strides_[2];

 f[(s) * 3 + 0] += v[0] * (1.0 - r[0]);
 f[(s - strides_[0]) * 3 + 0] += v[0] * (r[0]);
 f[(s) * 3 + 1] += v[1] * (1.0 - r[1]);
 f[(s - strides_[1]) * 3 + 1] += v[1] * (r[1]);
 f[(s) * 3 + 2] += v[2] * (1.0 - r[2]);
 f[(s - strides_[2]) * 3 + 2] += v[2] * (r[2]);

 }
 *
 *
 * */

#endif /* UNIFORM_RECT_OPS_H_ */

/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

#include <omp.h>
#include <vector>
#include <exception>
#include <numeric>

#include "include/simpla_defs.h"

#include "fetl/fetl.h"

#include "fetl/primitives.h"

#include "fetl/geometry.h"

#include "fetl/vector_calculus.h"

#include "grid.h"

namespace simpla
{
/**
 *  UniformRectGrid -- Uniform rectangular structured grid.
 * */
struct UniformRectGrid: public BaseGrid
{
	static const int NUM_OF_DIMS = 3;

	UniformRectGrid &
	operator=(const UniformRectGrid&);

	IVec3 shift_;

	std::vector<size_t> center_ele_[4];
	std::vector<size_t> ghost_ele_[4];

	typedef Real ValueType;
	typedef UniformRectGrid Grid;
	typedef TR1::shared_ptr<Grid> Holder;
	typedef size_t IndexType;
	typedef RVec3 CoordinatesType;
	typedef UniformRectGrid ThisType;

	enum
	{
		NDIMS = THREE
	};
	typedef typename std::vector<IndexType>::iterator iterator;
	typedef typename std::vector<IndexType>::const_iterator const_iterator;

	typedef RVec3 Coordinate;

	typedef TR1::shared_ptr<ByteType> Storage;

	Real dt;
	// Geometry
	RVec3 xmin, xmax;
	// Topology
	IVec3 dims;
	IVec3 gw;

	IVec3 strides;
	RVec3 inv_dx;
	RVec3 dx;

	UniformRectGrid() :
			dt(0)
	{
	}
	~UniformRectGrid()
	{
	}

	void SetGeometry(Real pdt, RVec3 pxmin, RVec3 pxmax, IVec3 pdims, IVec3 pgw)
	{
		dt = pdt;
		xmin = pxmin;
		xmax = pxmax;
		dims = pdims;
		gw = pgw;

		Init();
	}

	void Init()
	{
		for (int i = 0; i < NDIMS; ++i)
		{
			gw[i] = (gw[i] * 2 > dims[i]) ? dims[i] / 2 : gw[i];
			if (dims[i] <= 1)
			{
				dims[i] = 1;
				xmax[i] = xmin[i];
				dx[i] = 0.0;
				inv_dx[i] = 0.0;
			}
			else
			{
				dx[i] = (xmax[i] - xmin[i]) / static_cast<Real>(dims[i] - 1);
				inv_dx[i] = 1.0 / dx[i];
			}
		}

		strides[2] = 1;
		strides[1] = dims[2];
		strides[0] = dims[1] * dims[2];

		//#pragma omp parallel for  << here can not be parallized
		for (IndexType i = 0; i < dims[0]; ++i)
			for (IndexType j = 0; j < dims[1]; ++j)
				for (IndexType k = 0; k < dims[2]; ++k)
				{
					IndexType s = (i * strides[0] + j * strides[1]
							+ k * strides[2]);

					for (int f = 0; f < 4; ++f)
					{
						IndexType num_of_comp = get_num_of_comp(f);
						for (IndexType l = 0; l < num_of_comp; ++l)
						{
							if (i < gw[0] || i > dims[0] - gw[0] //
							|| j < gw[1] || j > dims[1] - gw[1] //
							|| k < gw[2] || k > dims[2] - gw[2])
							{
								ghost_ele_[f].push_back(s * num_of_comp + l);
							}
							else
							{

								center_ele_[f].push_back(s * num_of_comp + l);
							}
						}

					}
				}

	}

	template<int N>
	Geometry<ThisType, N> get_sub_geometry()
	{
		return (Geometry<ThisType, N>(*this));

	}

	inline std::vector<IndexType> const &
	get_center_elements(int iform) const
	{
		return (center_ele_[iform]);
	}

	inline std::vector<IndexType>::const_iterator get_center_elements_begin(
			int iform) const
	{
		return (center_ele_[iform].begin());
	}
	inline std::vector<IndexType>::const_iterator get_center_elements_end(
			int iform) const
	{
		return (center_ele_[iform].end());
	}
	inline std::vector<IndexType> const &
	get_ghost_elements(int iform) const
	{
		return (ghost_ele_[iform]);
	}

	inline bool operator==(Grid const & r) const
	{
		return (this == &r);
	}

// Property -----------------------------------------------

	inline IndexType get_num_of_vertex() const
	{
		IndexType res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 0) ? dims[i] : 1;
		}
		return (res);
	}
	inline IndexType get_num_of_edge() const
	{

		return (0);
	}
	inline IndexType get_num_of_face() const
	{
		return (0);
	}
	inline IndexType get_num_of_cell(int iform = 0) const
	{
		IndexType res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 1) ? (dims[i] - 1) : 1;
		}
		return (res);
	}

	inline RVec3 get_cell_center(IndexType s) const
	{
		//TODO UNIMPLEMENTED!!
		RVec3 res =
		{ 0, 0, 0 };
		return (res);
	}
	inline IndexType get_cell_num(IVec3 const & I) const
	{
		return (I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]);
	}
	inline IndexType get_cell_num(IndexType I0, IndexType I1,
			IndexType I2) const
	{
		return (I0 * strides[0] + I1 * strides[1] + I2 * strides[2]);
	}
	inline IndexType get_cell_num(RVec3 x) const
	{
		IVec3 I;
		I = (x - xmin) * inv_dx;
		return ((I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]));
	}

	inline Real get_cell_volumn(IndexType s = 0) const
	{
		Real res = 1.0;
		for (int i = 0; i < 3; ++i)
		{
			if (!isinf(dx[i]))
			{
				res *= (dims[i] - 1) * dx[i];
			}
		}

		return (res);
	}

	inline Real get_cell_d_volumn(IndexType s = 0) const
	{
		Real res = 1.0;
		for (int i = 0; i < 3; ++i)
		{
			if (!isinf(dx[i]) && dx[i] > 0)
			{
				res *= dx[i];
			}
		}

		return (res);
	}
	inline IndexType get_num_of_elements(int iform) const
	{
		return get_num_of_comp(iform) * get_num_of_vertex();

	}

	static inline IndexType get_num_of_comp(int iform)
	{
		static const int comps[4] =
		{ 1, 3, 3, 1 };

		return (comps[iform]);
	}

	inline std::vector<IndexType> get_field_shape(int iform) const
	{
		int ndims = 1;
//		FIXME (iform == 1 || iform == 2) ? NDIMS + 1 : NDIMS;

		std::vector<IndexType> d(ndims);
		for (int i = 0; i < NDIMS; ++i)
		{
			d[i] = dims[i];
		}
		if (iform == 1 || iform == 2)
		{
			d[NDIMS] = get_num_of_comp(iform);
		}
		return (d);
	}

// Assign Operation --------------------------------------------

	template<int IF, typename TV> inline //
	void InitEmptyField(Field<Geometry<Grid, IF>, TV> * f) const
	{
		if (f->storage == Storage())
		{
			f->storage =
					Storage(
							reinterpret_cast<ByteType*>(operator new(
									get_num_of_elements(IF)
											* (f->value_size_in_bytes))));
		}
	}

	template<int IF, typename TV> TV const & //
	GetConstValue(Field<Geometry<Grid, IF>, TV> const &f, IndexType s) const
	{
		return (*reinterpret_cast<const TV*>(&(*f.storage)
				+ s * f.value_size_in_bytes));
	}

	template<int IF, typename TV> TV & //
	GetValue(Field<Geometry<Grid, IF>, TV> &f, IndexType s) const
	{
		return (*reinterpret_cast<TV*>(&(*f.storage) + s * f.value_size_in_bytes));
	}

	template<int IFORM, typename TExpr, typename TR>
	void Assign(Field<Geometry<Grid, IFORM>, TExpr> & lhs, TR rhs) const
	{
		ASSERT(lhs.grid==*this);
		IndexType ele_num = get_num_of_elements(IFORM);

#pragma omp parallel for
		for (IndexType i = 0; i < ele_num; ++i)
		{
			lhs[i] = index(rhs, i);
		}
	}

// @NOTE the propose of this function is to assign constant vector to a field.
//   It confuses the semantics of nTuple with constant Field, and was discarded.
//	template<int IFORM, typename TExpr, int NR, typename TR>
//	void Assign(Field<Geometry<Grid, IFORM>, TExpr> & lhs, nTuple<NR, TR> rhs) const
//	{
//		ASSERT(lhs.grid==*this);
//		Index ele_num = get_num_of_elements(IFORM);
//
//#pragma omp parallel for
//		for (Index i = 0; i < ele_num; ++i)
//		{
//			lhs[i] = rhs[i % NR];
//		}
//	}
	template<int IFORM, typename TL, typename TR> void //
	Assign(Field<Geometry<Grid, IFORM>, TL>& lhs,
			Field<Geometry<Grid, IFORM>, TR> const& rhs) const
	{
		ASSERT(lhs.grid==*this);
		{
			std::vector<IndexType> const & ele_list = get_center_elements(
					IFORM);
			IndexType ele_num = ele_list.size();

#pragma omp parallel for
			for (IndexType i = 0; i < ele_num; ++i)
			{
				lhs[ele_list[i]] = rhs[ele_list[i]];
			}

		}
	}

//	template<int IFORM, typename TLExpr, typename TRExpr> inline auto //
//	InnerProduct(Field<Geometry<Grid, IFORM>, TLExpr> const & lhs,
//			Field<Geometry<Grid, IFORM>, TRExpr> const & rhs) const
//	{
//		typedef decltype(lhs[0] * rhs[0]) Value;
//
//		Value res;
//		res = 0;
//
//		std::vector<Index> const & ele_list = get_center_elements(IFORM);
//		Index ele_num = ele_list.size();
//
//#pragma omp parallel for reduction(+:res)
//		for (Index i = 0; i < ele_num; ++i)
//		{
//			res += lhs[ele_list[i]] * rhs[ele_list[i]];
//		}
//
//		return (res);
//
//	}

	template<int IFORM, typename TL, typename TR>
	static void //
	Add(Field<Geometry<Grid, IFORM>, TL> & lhs,
			Field<Geometry<Grid, IFORM>, TR> const& rhs)
	{
		if (lhs.grid == rhs.grid)
		{
			IndexType size = lhs.size();

			// NOTE this is parallelism of FDTD
#pragma omp parallel for
			for (IndexType s = 0; s < size; ++s)
			{
				lhs[s] += rhs[s];
			}
		}

		else
		{
			ERROR << "Grid mismatch!" << std::endl;
			throw(-1);
		}
	}

// Interpolation ----------------------------------------------------------

	template<typename TExpr>
	inline typename Field<Geometry<Grid, 0>, TExpr>::Value //
	Gather(Field<Geometry<Grid, 0>, TExpr> const &f, RVec3 x) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		return (f[s] * (1.0 - r[0]) + f[s + strides[0]] * r[0]); //FIXME Only for 1-dim
	}

	template<typename TExpr>
	inline void //
	Scatter(Field<Geometry<Grid, 0>, TExpr> & f, RVec3 x,
			typename Field<Geometry<Grid, 0>, TExpr>::Value const v) const
	{
		typename Field<Geometry<Grid, 0>, TExpr>::Value res;
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);
		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f.Add(s, v * (1.0 - r[0]));
		f.Add(s + strides[0], v * r[0]); //FIXME Only for 1-dim

	}

	template<typename TExpr>
	inline nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value>     //
	Gather(Field<Geometry<Grid, 1>, TExpr> const &f, RVec3 x) const
	{
		nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value> res;

		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx = r + 0.5;
		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		res[0] = (f[(s) * 3 + 0] * (0.5 - r[0])
				+ f[(s - strides[0]) * 3 + 0] * (0.5 + r[0]));
		res[1] = (f[(s) * 3 + 1] * (0.5 - r[1])
				+ f[(s - strides[1]) * 3 + 1] * (0.5 + r[1]));
		res[2] = (f[(s) * 3 + 2] * (0.5 - r[2])
				+ f[(s - strides[2]) * 3 + 2] * (0.5 + r[2]));
		return res;
	}
	template<typename TExpr>
	inline void //
	Scatter(Field<Geometry<Grid, 1>, TExpr> & f, RVec3 x,
			nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value> const &v) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx = r + 0.5;
		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
		f[(s - strides[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
		f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
		f[(s - strides[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
		f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
		f[(s - strides[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
	}

	template<typename TExpr>
	inline nTuple<THREE, typename Field<Geometry<Grid, 2>, TExpr>::Value>     //
	Gather(Field<Geometry<Grid, 2>, TExpr> const &f, RVec3 x) const
	{
		nTuple<THREE, typename Field<Geometry<Grid, 2>, TExpr>::Value> res;

		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		res[0] = (f[(s) * 3 + 0] * (1.0 - r[0])
				+ f[(s - strides[0]) * 3 + 0] * (r[0]));
		res[1] = (f[(s) * 3 + 1] * (1.0 - r[1])
				+ f[(s - strides[1]) * 3 + 1] * (r[1]));
		res[2] = (f[(s) * 3 + 2] * (1.0 - r[2])
				+ f[(s - strides[2]) * 3 + 2] * (r[2]));
		return res;

	}

	template<typename TExpr>
	inline void //
	Scatter(Field<Geometry<Grid, 2>, TExpr> & f, RVec3 x,
			nTuple<THREE, typename Field<Geometry<Grid, 2>, TExpr>::Value> const &v) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		IndexType s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f[(s) * 3 + 0] += v[0] * (1.0 - r[0]);
		f[(s - strides[0]) * 3 + 0] += v[0] * (r[0]);
		f[(s) * 3 + 1] += v[1] * (1.0 - r[1]);
		f[(s - strides[1]) * 3 + 1] += v[1] * (r[1]);
		f[(s) * 3 + 2] += v[2] * (1.0 - r[2]);
		f[(s - strides[2]) * 3 + 2] += v[2] * (r[2]);

	}

// Mapto ----------------------------------------------------------
	/**
	 *    mapto(Int2Type<0> ,   //target topology position
	 *     Field<Grid,1 , TExpr> const & vl,  //field
	 *      SizeType s   //grid index of point
	 *      )
	 * target topology position:
	 *             z 	y 	x
	 *       000 : 0,	0,	0   vertex
	 *       001 : 0,	0,1/2   edge
	 *       010 : 0, 1/2,	0
	 *       100 : 1/2,	0,	0
	 *       011 : 0, 1/2,1/2   Face
	 * */

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------
	template<typename TExpr> inline auto //
	eval(UniOp<OpGrad, TExpr> const & expr, IndexType const & s) const
	DECL_RET_TYPE(
			(expr.expr[(s - s % 3) / 3 + strides[s % 3]]
					- expr.expr[(s - s % 3) / 3]) * inv_dx[s % 3])

	template<typename TExpr> inline auto //
	eval(UniOp<OpDiverge, TExpr> const & expr,
			IndexType const & s) const
					DECL_RET_TYPE(

							(expr.expr[s * 3 + 0] - expr.expr[s * 3 + 0 - 3 * strides[0]])
							* inv_dx[0] +

							(expr.expr[s * 3 + 1] - expr.expr[s * 3 + 1 - 3 * strides[1]])
							* inv_dx[1] +

							(expr.expr[s * 3 + 2] - expr.expr[s * 3 + 2 - 3 * strides[2]])
							* inv_dx[2]
					)
	template<typename TL> inline auto //
	eval(UniOp<OpCurl, TL> const & expr,
			IndexType const & s) const
					DECL_RET_TYPE(
							(expr.expr[s - s %3 + (s + 2) % 3 + 3 * strides[(s + 1) % 3]] - expr.expr[s - s %3 + (s + 2) % 3])
							* inv_dx[(s + 1) % 3]-

							(expr.expr[s - s %3 + (s + 1) % 3 + 3 * strides[(s + 2) % 3]] - expr.expr[s - s %3 + (s + 1) % 3])
							* inv_dx[(s + 2) % 3]
					)

	template<typename TL> auto // Field<Geometry<Grid, 1>,
	eval(UniOp<OpCurl, TL> const & expr,
			IndexType const & s) const
					DECL_RET_TYPE(

							(expr.expr[s - s % 3 + (s + 2) % 3] - expr.expr[s - s % 3 + (s + 2) % 3 - 3 * strides[(s + 1) % 3]])
							* inv_dx[(s + 1) % 3]-

							(expr.expr[s - s % 3 + (s + 1) % 3] - expr.expr[s - s % 3 + (s + 1) % 3 - 3 * strides[(s + 1) % 3]])
							* inv_dx[(s + 2) % 3]
					)

	template<int IPD, typename TExpr> inline auto // Field<Geometry<Grid, 1>,
	eval(UniOp<OpCurlPD<IPD>, TExpr> const & expr,
			IndexType const &s) const ->
			typename std::enable_if<order_of_form<TExpr>::value==1, decltype(expr[0]) >::type
	{
		if (dims[IPD] == 1)
		{
			return (0);
		}
		IndexType j0 = s % 3;

		IndexType idx1 = s - j0;
		typename TExpr::Value res = 0.0;
		if (1 == IPD)
		{
			res = (expr.rhs_[idx1 + 2 + 3 * strides[IPD]] - expr.rhs_[idx1 + 2])
					* inv_dx[IPD];
		}
		else if (2 == IPD)
		{
			res =
					(-expr.rhs_[idx1 + 1 + 3 * strides[IPD]]
							+ expr.rhs_[idx1 + 1]) * inv_dx[IPD];
		}

		return (res);
	}

	template<int IPD, typename TExpr> inline auto //	Field<Geometry<Grid, 2>,
	eval(UniOp<OpCurlPD<IPD>, TExpr> const & expr,
			IndexType const &s) const ->
			typename std::enable_if<order_of_form<TExpr>::value==2, decltype(expr[0]) >::type
	{
		if (dims[IPD] == 1)
		{
			return (0);
		}
		IndexType j0 = s % 3;

		IndexType idx2 = s - j0;

		typename Field<Geometry<Grid, 2>, TExpr>::Value res = 0.0;
//		if (1 == IPD)
//		{
//			res = (expr.rhs_[idx2 + 2]
//					- expr.rhs_[idx2 + 2 - 3 * strides[IPD]]) * inv_dx[IPD];
//
//		}
//		else if (2 == IPD)
//		{
//			res = (-expr.rhs_[idx2 + 1]
//					+ expr.rhs_[idx2 + 1 - 3 * strides[IPD]]) * inv_dx[IPD];
//		}

		return (res);
	}

};

} //namespace simpla
#endif //UNIFORM_RECT_H_

/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */

#ifndef DEFAULT_GRID

#define DEFAULT_GRID UniformRectGrid

#include <omp.h>
#include <vector>
#include <exception>
#include <numeric>

#include "include/simpla_defs.h"
#include "primitives/ntuple.h"
#include "fetl/fetl_defs.h"
#include "engine/object.h"

namespace simpla
{
namespace fetl
{

/**
 *  UniformRectGrid -- Uniform rectangular structured grid.
 * */
class UniformRectGrid:public Object
{

	UniformRectGrid & operator=(const UniformRectGrid&);
	bool initialized_;

	UniformRectGrid const * parent_;
	IVec3 shift_;

	std::vector<size_t> center_ele_[4];
	std::vector<size_t> ghost_ele_[4];
public:
	typedef UniformRectGrid ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	static const int NDIMS = THREE;
	typedef typename std::vector<size_t>::iterator iterator;
	typedef typename std::vector<size_t>::const_iterator const_iterator;
	Real dt;
	Real time;
	Real counter;
	// Geometry
	Vec3 xmin, xmax;
	Vec3 dx;
	Vec3 inv_dx;
	// Topology
	IVec3 dims;
	IVec3 strides;
	IVec3 gw;

	UniformRectGrid(const UniformRectGrid &rhs)
	{
		;
	}

	UniformRectGrid() :
			initialized_(false)
	{
	}
	~UniformRectGrid()
	{
	}

	void Initialize(Real _dt, Vec3 _xmin, Vec3 _xmax, IVec3 _dims)
	{
		dt = _dt;
		gw[0] = 2;
		gw[1] = 2;
		gw[2] = 2;

		for (int i = 0; i < NDIMS; ++i)
		{
			dims[i] = _dims[i];
			xmin[i] = _xmin[i];
			xmax[i] = _xmax[i];
			gw[i] = (gw[i] * 2 > dims[i]) ? dims[i] / 2 : gw[i];
		}

		for (int i = 0; i < NDIMS; ++i)
		{
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

		initialized_ = true;

//#pragma omp parallel for  << here can not be parallized
		for (size_t i = 0; i < dims[0]; ++i)
			for (size_t j = 0; j < dims[1]; ++j)
				for (size_t k = 0; k < dims[2]; ++k)
				{
					size_t s =
							(i * strides[0] + j * strides[1] + k * strides[2]);

					for (int f = 0; f < 4; ++f)
					{
						size_t num_of_comp = get_num_of_comp(f);
						for (size_t l = 0; l < num_of_comp; ++l)
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

	inline std::vector<size_t> const & get_center_elements(int iform) const
	{
		return (center_ele_[iform]);
	}

	inline std::vector<size_t>::const_iterator get_center_elements_begin(
			int iform) const
	{
		return (center_ele_[iform].begin());
	}
	inline std::vector<size_t>::const_iterator get_center_elements_end(
			int iform) const
	{
		return (center_ele_[iform].end());
	}
	inline std::vector<size_t> const & get_ghost_elements(int iform) const
	{
		return (ghost_ele_[iform]);
	}

	inline bool operator==(ThisType const & r) const
	{
		return (this == &r);
	}
	ThisType const & parent() const
	{
		return (*parent_);
	}

	ThisType SubGrid(size_t s, IVec3 w) const
	{
		IVec3 ix0;
		IVec3 imin, imax;
		imin = ix0 - w;
		imin = ix0 + w;
		return SubGrid(imin, imax);
	}

	ThisType SubGrid(RVec3 x, RVec3 w) const
	{
		IVec3 imin, imax;
		imin = (x - w) * inv_dx;
		imax = (x + w) * inv_dx;
		return SubGrid(imin, imax);
	}

	ThisType SubGrid(RVec3 x, IVec3 w) const
	{
		IVec3 ix0;
		ix0 = x * inv_dx;
		IVec3 imin, imax;
		imin = ix0 - w;
		imin = ix0 + w;
		return SubGrid(imin, imax);
	}

	ThisType SubGrid(IVec3 imin, IVec3 imax) const
	{
		ThisType res;
		Vec3 pxmin, pxmax;
		pxmin = imin * dx;
		pxmax = imax * dx;
		IVec3 pdims;
		pdims = xmax - xmin;

		res.Initialize(dt, pxmin, pxmax, pdims);

		res.parent_ = this;
		res.shift_ = imin;
		return res;
	}

	// Property -----------------------------------------------

	inline size_t get_num_of_vertex() const
	{
		size_t res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 0) ? dims[i] : 1;
		}
		return (res);
	}
	inline size_t get_num_of_edge() const
	{

		return (0);
	}
	inline size_t get_num_of_face() const
	{
		return (0);
	}
	inline size_t get_num_of_cell(int iform = 0) const
	{
		size_t res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 1) ? (dims[i] - 1) : 1;
		}
		return (res);
	}

	inline RVec3 get_cell_center(size_t s) const
	{
		//TODO UNIMPLEMENTED!!
		RVec3 res;
		return (res);
	}
	inline size_t get_cell_num(size_t I0, size_t I1, size_t I2) const
	{
		return (I0 * strides[0] + I1 * strides[1] + I2 * strides[2]);
	}
	inline size_t get_cell_num(RVec3 x) const
	{
		IVec3 I;
		I = (x - xmin) * inv_dx;
		return ((I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]));
	}

	inline Real get_cell_volumn(size_t s = 0) const
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

	inline Real get_cell_d_volumn(size_t s = 0) const
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
	inline size_t get_num_of_elements(int iform) const
	{
		return get_num_of_comp(iform) * get_num_of_vertex();

	}

	static inline size_t get_num_of_comp(int iform)
	{
		int res = 0;

		switch (iform)
		{
		case IZeroForm:
			res = 1;
			break;
		case IOneForm:
			res = 3;
			break;
		case ITwoForm:
			res = 3;
			break;
		case IThreeForm:
			res = 1;
			break;
		}
		return (res);
	}

	inline int get_field_shape(size_t *d, int iform) const
	{
		int ndims =
				(iform == IOneForm || iform == ITwoForm) ? NDIMS + 1 : NDIMS;
		if (d != NULL)
		{
			for (int i = 0; i < NDIMS; ++i)
			{
				d[i] = dims[i];
			}
			if (iform == IOneForm || iform == ITwoForm)
			{
				d[NDIMS] = get_num_of_comp(iform);
			}
		}
		return ndims;
	}

	// Assign Operation --------------------------------------------

//	template<int IFORM, typename TV, typename TL, int N, typename TR> void //
//	Assign(Field<IFORM, TV, TL> & lhs, nTuple<N, TR> rhs) const
//	{
//		ASSERT(lhs.grid==*this);
//		size_t ele_num = get_num_of_elements(Field<IFORM, TV, TL>::IForm);
//
//#pragma omp parallel for
//		for (size_t i = 0; i < ele_num; ++i)
//		{
//			lhs[i] = rhs[i % N];
//
//		}
//	}

	template<int IFORM, typename TV, typename TL, typename TRV> void //
	Assign(Field<IFORM, TV, TL> & lhs, nTuple<THREE, TRV> const & rhs) const
	{
		ASSERT(lhs.grid==*this);
		size_t ele_num = get_num_of_elements(Field<IFORM, TV, TL>::IForm);

#pragma omp parallel for
		for (size_t i = 0; i < ele_num; ++i)
		{
			lhs[i] = rhs[3];
		}
	}

	template<int IFORM, typename TV, typename TL, typename TR> void //
	Assign(Field<IFORM, TV, TL> & lhs,TR rhs) const
	{
		ASSERT(lhs.grid==*this);
		size_t ele_num = get_num_of_elements(Field<IFORM, TV, TL>::IForm);

#pragma omp parallel for
		for (size_t i = 0; i < ele_num; ++i)
		{
			lhs[i] = rhs;
		}
	}

	template<int IFORM, typename TV, typename TL, typename TR> void //
	Assign(Field<IFORM, nTuple<THREE, TV> , TL> & lhs
			,nTuple<THREE, TR> rhs) const
	{
		ASSERT(lhs.grid==*this);
		size_t ele_num = get_num_of_elements(Field<IFORM, TV, TL>::IForm);

#pragma omp parallel for
		for (size_t i = 0; i < ele_num; ++i)
		{
			lhs[i] = rhs;

		}
	}

	template<int IFORM, typename TV, typename TL, typename TVR, typename TR> void //
	Assign(Field<IFORM, TV, TL>& lhs, Field<IFORM, TVR, TR> const& rhs) const
	{
		ASSERT(lhs.grid==*this);
		if (lhs.grid == rhs.grid)
		{
			std::vector<size_t> const & ele_list = get_center_elements(
					Field<IFORM, TV, TL>::IForm);
			size_t ele_num = ele_list.size();
#pragma omp parallel for
			for (size_t i = 0; i < ele_num; ++i)
			{
				lhs[ele_list[i]] = rhs[ele_list[i]];
			}

		}
//		else if (lhs.grid == rhs.grid.parent())
//		{
//#pragma omp parallel for
//			for (size_t i = 0; i < rhs.grid.dims[0]; ++i)
//				for (size_t j = 0; j < rhs.grid.dims[1]; ++j)
//					for (size_t k = 0; k < rhs.grid.dims[2]; ++k)
//						for (size_t l = 0;
//								l < lhs.grid.get_num_of_comp(lhs.IForm); ++l)
//						{
//							lhs[
//
//							(i + rhs.grid.shift_[0]) * lhs.grid.strides[0] +
//
//							(j + rhs.grid.shift_[1]) * lhs.grid.strides[1] +
//
//							(k + rhs.grid.shift_[2]) * lhs.grid.strides[2] + l
//
//							]
//
//							=
//
//							rhs[
//
//							i * rhs.grid.strides[0] +
//
//							j * rhs.grid.strides[1] +
//
//							k * rhs.grid.strides[2] + l
//
//							];
//						}
//		}
//		else if (lhs.grid.parent() == rhs.grid)
//		{
//
//#pragma omp parallel for
//			for (size_t i = 0; i < lhs.grid.dims[0]; ++i)
//				for (size_t j = 0; j < lhs.grid.dims[1]; ++j)
//					for (size_t k = 0; k < lhs.grid.dims[2]; ++k)
//						for (size_t l = 0;
//								l < lhs.grid.get_num_of_comp(lhs.IForm); ++l)
//						{
//							lhs[
//
//							i * lhs.grid.strides[0] +
//
//							j * lhs.grid.strides[1] +
//
//							k * lhs.grid.strides[2] + l]
//
//							=
//
//							rhs[
//
//							(i + lhs.grid.shift_[0]) * rhs.grid.strides[0] +
//
//							(j + lhs.grid.shift_[1]) * rhs.grid.strides[1] +
//
//							(k + lhs.grid.shift_[2]) * rhs.grid.strides[2] + l];
//						}
//		}
		else
		{
			ERROR<<"Grid mismatch!" << std::endl;
			throw(-1);
		}
	}

	template<int IFORM, typename TVL, typename TLExpr, typename TVR,
			typename TRExpr>
	typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType //
	InnerProduct(Field<IFORM, TVL, TLExpr> const & lhs
			, Field<IFORM, TVR, TRExpr> const & rhs) const
	{
		typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
		ValueType res;
		res = 0;

		std::vector<size_t> const & ele_list = get_center_elements(IFORM);
		size_t ele_num = ele_list.size();

#pragma omp parallel for reduction(+:res)
		for (size_t i = 0; i < ele_num; ++i)
		{
			res += lhs[ele_list[i]] * rhs[ele_list[i]];
		}

		return (res);

	}

	template<int IFORM, typename TV, typename TExpr> static void //
	Add(Field<IFORM, TV, TExpr> & lhs, const TV & rhs)
	{
		size_t size = lhs.size();

		// NOTE this is the parallelism of FDTD
#pragma omp parallel for
		for (size_t s = 0; s < size; ++s)
		{
			lhs[s] = mapto_(Int2Type<IFORM>(), rhs, s);
		}
	}

	template<int IFORM, typename TV, typename TL, typename TR> static void //
	Add(Field<IFORM, TV, TL> & lhs, Field<IFORM, TV, TR> const& rhs)
	{
		if (lhs.grid == rhs.grid)
		{
			size_t size = lhs.size();

			// NOTE this is parallelism of FDTD
#pragma omp parallel for
			for (size_t s = 0; s < size; ++s)
			{
				lhs[s] += rhs[s];
			}
		}
		else if (lhs.grid == rhs.grid.parent())
		{
#pragma omp parallel for
			for (size_t i = 0; i < rhs.grid.dims[0]; ++i)
				for (size_t j = 0; j < rhs.grid.dims[1]; ++j)
					for (size_t k = 0; k < rhs.grid.dims[2]; ++k)
						for (size_t l = 0; l < get_num_of_comp(lhs.IForm); ++l)
						{
							lhs[

							(i + rhs.grid.shift_[0]) * lhs.grid.strides[0] +

							(j + rhs.grid.shift_[1]) * lhs.grid.strides[1] +

							(k + rhs.grid.shift_[2]) * lhs.grid.strides[2] + l

							]

							+=

							rhs[

							i * rhs.grid.strides[0] +

							j * rhs.grid.strides[1] +

							k * rhs.grid.strides[2] + l

							];
						}
		}
		else if (lhs.grid.parent() == rhs.grid)
		{

			// NOTE this is parallelism of FDTD
#pragma omp parallel for
			for (size_t i = 0; i < lhs.grid.dims[0]; ++i)
				for (size_t j = 0; j < lhs.grid.dims[1]; ++j)
					for (size_t k = 0; k < lhs.grid.dims[2]; ++k)
						for (size_t l = 0; l < get_num_of_comp(lhs.IForm); ++l)
						{
							lhs[

							i * lhs.grid.strides[0] +

							j * lhs.grid.strides[1] +

							k * lhs.grid.strides[2] + l]

							+=

							rhs[

							(i + lhs.grid.shift_[0]) * rhs.grid.strides[0] +

							(j + lhs.grid.shift_[1]) * rhs.grid.strides[1] +

							(k + lhs.grid.shift_[2]) * rhs.grid.strides[2] + l];
						}
		}
		else
		{
			ERROR<<"Grid mismatch!" << std::endl;
			throw(-1);
		}
	}

	// Interpolation ----------------------------------------------------------

	template<typename TV, typename TExpr> inline TV //
	Gather(Field<IZeroForm, TV, TExpr> const &f, RVec3 x) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		return (f[s] * (1.0 - r[0]) + f[s + strides[0]] * r[0]); //FIXME Only for 1-dim
	}

	template<typename TV, typename TExpr> inline void //
	Scatter(Field<IZeroForm, TV, TExpr> & f, RVec3 x, TV const v) const
	{
		TV res;
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);
		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];
//		CHECK(x);
//		CHECK(r);
//		CHECK(inv_dx);
//		CHECK(xmin);
//		CHECK(idx);
		f.Add(s, v * (1.0 - r[0]));
		f.Add(s + strides[0], v * r[0]); //FIXME Only for 1-dim

	}

	template<typename TV, typename TExpr> inline nTuple<THREE, TV>  //
	Gather(Field<IOneForm, TV, TExpr> const &f, RVec3 x) const
	{
		nTuple<THREE, TV> res;

		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx = r + 0.5;
		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		res[0] = (f[(s) * 3 + 0] * (0.5 - r[0])
				+ f[(s - strides[0]) * 3 + 0] * (0.5 + r[0]));
		res[1] = (f[(s) * 3 + 1] * (0.5 - r[1])
				+ f[(s - strides[1]) * 3 + 1] * (0.5 + r[1]));
		res[2] = (f[(s) * 3 + 2] * (0.5 - r[2])
				+ f[(s - strides[2]) * 3 + 2] * (0.5 + r[2]));
		return res;
	}
	template<typename TV, typename TExpr> inline void //
	Scatter(Field<IOneForm, TV, TExpr> & f, RVec3 x
			, nTuple<THREE, TV> const &v) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx = r + 0.5;
		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
		f[(s - strides[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
		f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
		f[(s - strides[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
		f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
		f[(s - strides[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
	}

	template<typename TV, typename TExpr> inline nTuple<THREE, TV>  //
	Gather(Field<ITwoForm, TV, TExpr> const &f, RVec3 x) const
	{
		nTuple<THREE, TV> res;

		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		res[0] = (f[(s) * 3 + 0] * (1.0 - r[0])
				+ f[(s - strides[0]) * 3 + 0] * (r[0]));
		res[1] = (f[(s) * 3 + 1] * (1.0 - r[1])
				+ f[(s - strides[1]) * 3 + 1] * (r[1]));
		res[2] = (f[(s) * 3 + 2] * (1.0 - r[2])
				+ f[(s - strides[2]) * 3 + 2] * (r[2]));
		return res;

	}

	template<typename TV, typename TExpr> inline void //
	Scatter(Field<ITwoForm, TV, TExpr> & f, RVec3 x
			, nTuple<THREE, TV> const &v) const
	{
		IVec3 idx;
		Vec3 r;
		r = (x - xmin) * inv_dx;
		idx[0] = static_cast<long>(r[0]);
		idx[1] = static_cast<long>(r[1]);
		idx[2] = static_cast<long>(r[2]);

		r -= idx;
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
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
	 *    mapto(Int2Type<IZeroForm> ,   //target topology position
	 *     Field<IOneForm , TExpr> const & vl,  //field
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

////-----------------------------------------
//// map to
////-----------------------------------------
	template<int IForm, typename TV, typename TIter> inline TV //
	mapto_(Int2Type<IForm> const&, TV const & v, TIter) const
	{
		return v;
	}

	template<int IForm, int N, typename TV, typename TIter> inline TV // for constant vector
	mapto_(Int2Type<IForm> const&, nTuple<N, TV> const & v
			, TIter const &s) const
	{
		return (mapto_(Int2Type<IForm>(), v, s));
	}

	template<int IForm, int IFORMR, typename TV, typename TExpr, typename TIter> //
	inline typename Field<IFORMR, TV, TExpr>::ValueType // for fields transformation
	mapto_(Int2Type<IForm> const&, const Field<IFORMR, TV, TExpr> & expr ,
	TIter const & s) const
	{
		return (mapto_(Int2Type<IForm>(), Int2Type<IFORMR>(), expr, s));
	}
	template<int IForm, int N, typename TV> inline TV //
	mapto_(Int2Type<IForm>, nTuple<N, TV> const & v, size_t s) const
	{
		return v[s % N];
	}

	template<int IForm, typename TExpr> inline typename TExpr::ValueType // for same type field
	mapto_(Int2Type<IForm>, Int2Type<IForm>, const TExpr & expr, size_t s) const
	{
		return expr[s];
	}

	template<typename TExpr> inline typename TExpr::ValueType //
	mapto_(Int2Type<IOneForm>, Int2Type<IZeroForm>, const TExpr & expr,
			size_t s) const
	{
		size_t j0 = s % 3;
		size_t idx = (s - j0) / 3;
		return ((expr[idx] + expr[idx + strides[j0]]) * 0.5);
	}

	template<typename TExpr> inline typename TExpr::ValueType //
	mapto_(Int2Type<ITwoForm>, Int2Type<IZeroForm>, const TExpr & expr,
			size_t s) const
	{
		size_t j0 = s % 3;
		size_t j1 = (s + 1) % 3;
		size_t j2 = (s + 2) % 3;
		size_t idx = (s - j0) / 3;
		return ((expr[idx] + expr[idx + strides[j1]] + expr[idx + strides[j2]]
				+ expr[idx + strides[j1] + strides[j2]]) * 0.25);
	}

//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<IVecZeroForm>, Int2Type<IZeroForm>, const TExpr & expr,
//			size_t s) const
//	{
//		size_t j0 = s % 3;
//		size_t idx = (s - j0) / 3;
//		return (expr[idx]);
//	}

	template<typename TExpr> inline typename TExpr::ValueType //
	mapto_(Int2Type<IZeroForm>, Int2Type<IOneForm>, const TExpr & expr,
			size_t s) const
	{
		unsigned IS = s % 3;
		return ((expr[s] + expr[s - 3 * strides[s % 3]]) * 0.5);
	}

	template<typename TExpr> inline typename TExpr::ValueType //
	mapto_(Int2Type<IZeroForm>, Int2Type<ITwoForm>, const TExpr & expr,
			size_t s) const
	{
		return ((expr[s] + expr[s - 3 * strides[(s + 1) % 3]]
				+ expr[s - 3 * strides[(s + 2) % 3]]
				+ expr[s - 3 * strides[(s + 1) % 3] - 3 * strides[(s + 2) % 3]])
				* 0.25);
	}

//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<IZeroForm>, Int2Type<IVecZeroForm>, const TExpr & expr,
//			size_t s) const
//	{
//		return (expr[s]);
//	}

//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<IOneForm>, Int2Type<IVecZeroForm>, const TExpr & expr,
//			size_t s) const
//	{
//		size_t j0 = s % 3;
//		return ((expr[s] + expr[s + 3 * strides[j0]]) * 0.5);
//	}
//
//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<ITwoForm>, Int2Type<IVecZeroForm>, const TExpr & expr,
//			size_t s) const
//	{
//		size_t j0 = s % 3;
//		size_t j1 = (j0 + 1) % 3;
//		size_t j2 = (j0 + 2) % 3;
//		return (expr[s] + expr[s + 3 * strides[j1]] + expr[s + 3 * strides[j2]]
//				+ expr[s + 3 * strides[j1] + 3 * strides[j2]]) * 0.25;
//	}
//
//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<IVecZeroForm>, Int2Type<IOneForm>, const TExpr & expr,
//			size_t s) const
//	{
//		size_t j0 = s % 3;
//		return (expr[s - 3 * strides[j0]] + expr[s]) * 0.5;
//	}
//
//	template<typename TExpr> inline typename TExpr::ValueType //
//	mapto_(Int2Type<IVecZeroForm>, Int2Type<ITwoForm>, const TExpr & expr,
//			size_t s) const
//	{
//		size_t j0 = s % 3;
//		size_t j1 = (j0 + 1) % 3;
//		size_t j2 = (j0 + 2) % 3;
//		return (expr[s] + expr[s - 3 * strides[j1]] + expr[s - 3 * strides[j2]]
//				+ expr[s - 3 * strides[j1] - 3 * strides[j2]]) * 0.25;
//	}
//
//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------
	template<typename TExpr> inline typename TExpr::ValueType //
	dot_(TExpr const & expr, size_t s) const
	{
		return

		mapto_(Int2Type<IZeroForm>(), expr.lhs_, s * 3 + 0) *

		mapto_(Int2Type<IZeroForm>(), expr.rhs_, s * 3 + 0) +

		mapto_(Int2Type<IZeroForm>(), expr.lhs_, s * 3 + 1) *

		mapto_(Int2Type<IZeroForm>(), expr.rhs_, s * 3 + 1) +

		mapto_(Int2Type<IZeroForm>(), expr.lhs_, s * 3 + 2) *

		mapto_(Int2Type<IZeroForm>(), expr.rhs_, s * 3 + 2);
	}

	template<typename TV, typename TExpr> inline typename TExpr::ValueType //
	cross_(TExpr const & expr, size_t s) const
	{
		size_t j0 = s % 3;
		size_t j1 = (j0 + 1) % 3;
		size_t j2 = (j0 + 2) % 3;
		size_t idx = s - j0;

		return

		mapto_(Int2Type<TExpr::IForm>(), expr.lhs_, idx + j1) *

		mapto_(Int2Type<TExpr::IForm>(), expr.rhs_, idx + j2) -

		mapto_(Int2Type<TExpr::IForm>(), expr.lhs_, idx + j2) *

		mapto_(Int2Type<TExpr::IForm>(), expr.rhs_, idx + j1);

	}

	template<typename TV, typename TExpr> inline TV //
	grad_(Field<IZeroForm, TV, TExpr> const & expr, size_t s) const
	{
		size_t j0 = s % 3;
//		size_t j1 = (j0 + 1) % 3;
//		size_t j2 = (j0 + 2) % 3;
		size_t idx0 = (s - j0) / 3;
		return (expr[idx0 + strides[j0]] - expr[idx0]) * inv_dx[j0];
	}
	template<typename TV, typename TExpr> inline TV //
	diverge_(Field<IOneForm, TV, TExpr> const & expr, size_t s) const
	{
		return

		(expr[s * 3 + 0] - expr[s * 3 + 0 - 3 * strides[0]]) * inv_dx[0] +

		(expr[s * 3 + 1] - expr[s * 3 + 1 - 3 * strides[1]]) * inv_dx[1] +

		(expr[s * 3 + 2] - expr[s * 3 + 2 - 3 * strides[2]]) * inv_dx[2]

		;
	}

	template<typename TV, typename TExpr> inline TV //
	curl_(Field<IOneForm, TV, TExpr> const & expr, size_t s) const
	{
		size_t j0 = s % 3;
		size_t j1 = (j0 + 1) % 3;
		size_t j2 = (j0 + 2) % 3;
		size_t idx1 = s - j0;
		return

		(expr[idx1 + j2 + 3 * strides[j1]] - expr[idx1 + j2]) * inv_dx[j1] -

		(expr[idx1 + j1 + 3 * strides[j2]] - expr[idx1 + j1]) * inv_dx[j2];
	}
	template<typename TV, typename TExpr> inline TV //
	curl_(Field<ITwoForm, TV, TExpr> const & expr, size_t s) const
	{
		size_t j0 = s % 3;
		size_t j1 = (j0 + 1) % 3;
		size_t j2 = (j0 + 2) % 3;
		size_t idx2 = s - j0;
		return

		(expr[idx2 + j2] - expr[idx2 + j2 - 3 * strides[j1]]) * inv_dx[j1]

		- (expr[idx2 + j1] - expr[idx2 + j1 - 3 * strides[j2]]) * inv_dx[j2];
	}

	template<int PD, typename TV, typename TExpr> inline TV //
	curlPd_(Field<IOneForm, TV, TExpr> const & expr, size_t s) const
	{
		if (dims[PD] == 1)
		{
			return (0);
		}
		size_t j0 = s % 3;
		size_t j1 = (s + 1) % 3;
		size_t j2 = (s + 2) % 3;
		size_t idx1 = s - j0;
		TV res = 0.0;
		if (j1 == PD)
		{
			res = (expr[idx1 + j2 + 3 * strides[PD]] - expr[idx1 + j2])
					* inv_dx[PD];
		}
		else if (j2 == PD)
		{
			res = (-expr[idx1 + j1 + 3 * strides[PD]] + expr[idx1 + j1])
					* inv_dx[PD];
		}

		return (res);
	}

	template<int PD, typename TV, typename TExpr> inline TV //
	curlPd_(Field<ITwoForm, TV, TExpr> const & expr, size_t s) const
	{
		if (dims[PD] == 1)
		{
			return (0);
		}
		size_t j0 = s % 3;
		size_t j1 = (s + 1) % 3;
		size_t j2 = (s + 2) % 3;
		size_t idx2 = s - j0;

		TV res = 0.0;
		if (j1 == PD)
		{
			res = (expr[idx2 + j2] - expr[idx2 + j2 - 3 * strides[PD]])
					* inv_dx[PD];

		}
		else if (j2 == PD)
		{
			res = (-expr[idx2 + j1] + expr[idx2 + j1 - 3 * strides[PD]])
					* inv_dx[PD];
		}

		return (res);
	}
};

} // namespace fetl
} //namespace simpla
#endif  // DEFAULT_GRID

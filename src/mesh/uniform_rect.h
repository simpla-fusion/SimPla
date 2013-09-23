/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

#include <vector>
#include <exception>
#include <numeric>

#include "include/simpla_defs.h"

#include "fetl/primitives.h"

namespace simpla
{
template<typename, typename > class Field;
template<typename, int> class Geometry;
/**
 *  UniformRectMesh -- Uniform rectangular structured grid.
 * */
struct UniformRectMesh: public BaseMesh
{
	static const int NUM_OF_DIMS = 3;

	UniformRectMesh &
	operator=(const UniformRectMesh&);

	IVec3 shift_;

	std::vector<size_t> center_ele_[4];
	std::vector<size_t> ghost_ele_[4];

	typedef Real ValueType;
	typedef UniformRectMesh Grid;
	typedef std::shared_ptr<Grid> Holder;
	typedef RVec3 CoordinatesType;
	typedef UniformRectMesh ThisType;

	enum
	{
		NDIMS = THREE
	};
	typedef typename std::vector<size_t>::iterator iterator;
	typedef typename std::vector<size_t>::const_iterator const_iterator;

	typedef RVec3 Coordinate;

	typedef std::shared_ptr<ByteType> Storage;

	Real dt;
	// Geometry
	RVec3 xmin, xmax;
	// Topology
	IVec3 dims;
	IVec3 gw;

	IVec3 strides;
	RVec3 inv_dx;
	RVec3 dx;

	UniformRectMesh()
	{
	}
	~UniformRectMesh()
	{
	}

	template<typename TCONFIG>
	void Config(TCONFIG const & vm)
	{
		vm["dt"].get(dt);
		vm["xmin"].get(xmin);
		vm["xmax"].get(xmax);
		vm["dims"].get(dims);
		vm["gw"].get(gw);

		Init();
	}

	std::string Summary() const
	{
		return ("Coder is too lazy to implement it!");
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

	template<int N>
	Geometry<ThisType, N> get_sub_geometry()
	{
		return (Geometry<ThisType, N>(*this));

	}

	inline std::vector<size_t> const &
	get_center_elements(int iform) const
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

	inline size_t get_num_of_center_elements(int iform) const
	{
		return (center_ele_[iform].size());
	}

	inline std::vector<size_t> const &
	get_ghost_elements(int iform) const
	{
		return (ghost_ele_[iform]);
	}

	inline bool operator==(Grid const & r) const
	{
		return (this == &r);
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
		RVec3 res =
		{ 0, 0, 0 };
		return (res);
	}
	inline size_t get_cell_num(IVec3 const & I) const
	{
		return (I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]);
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
		static const int comps[4] =
		{ 1, 3, 3, 1 };

		return (comps[iform]);
	}

	inline std::vector<size_t> get_field_shape(int iform) const
	{
		int ndims = 1;
//		FIXME (iform == 1 || iform == 2) ? NDIMS + 1 : NDIMS;

		std::vector<size_t> d(ndims);
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
	GetConstValue(Field<Geometry<Grid, IF>, TV> const &f, size_t s) const
	{
		return (*reinterpret_cast<const TV*>(&(*f.storage)
				+ s * f.value_size_in_bytes));
	}

	template<int IF, typename TV> TV & //
	GetValue(Field<Geometry<Grid, IF>, TV> &f, size_t s) const
	{
		return (*reinterpret_cast<TV*>(&(*f.storage) + s * f.value_size_in_bytes));
	}

	template<int IFORM, typename TExpr, typename TR>
	void Assign(Field<Geometry<Grid, IFORM>, TExpr> & lhs,
			Field<Geometry<Grid, IFORM>, TR> const & rhs) const
	{
		size_t ele_num = get_num_of_elements(IFORM);

		for (size_t i = 0; i < ele_num; ++i)
		{
			lhs[i] = rhs[i];
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
//	template<int IFORM, typename TL, typename TR> void //
//	Assign(Field<Geometry<Grid, IFORM>, TL>& lhs,
//			Field<Geometry<Grid, IFORM>, TR> const& rhs) const
//	{
//		ASSERT(lhs.grid==*this);
//		{
//			std::vector<size_t> const & ele_list = get_center_elements(IFORM);
//			size_t ele_num = ele_list.size();
//
//#pragma omp parallel for
//			for (size_t i = 0; i < ele_num; ++i)
//			{
//				lhs[ele_list[i]] = rhs[ele_list[i]];
//			}
//
//		}
//	}

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
			size_t size = lhs.size();

			// NOTE this is parallelism of FDTD
#pragma omp parallel for
			for (size_t s = 0; s < size; ++s)
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
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
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
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f.Add(s, v * (1.0 - r[0]));
		f.Add(s + strides[0], v * r[0]); //FIXME Only for 1-dim

	}

	template<typename TExpr>
	inline nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value>    //
	Gather(Field<Geometry<Grid, 1>, TExpr> const &f, RVec3 x) const
	{
		nTuple<THREE, typename Field<Geometry<Grid, 1>, TExpr>::Value> res;

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
		size_t s = idx[0] * strides[0] + idx[1] * strides[1]
				+ idx[2] * strides[2];

		f[(s) * 3 + 0] += v[0] * (0.5 - r[0]);
		f[(s - strides[0]) * 3 + 0] += v[0] * (0.5 + r[0]);
		f[(s) * 3 + 1] += v[1] * (0.5 - r[1]);
		f[(s - strides[1]) * 3 + 1] += v[1] * (0.5 + r[1]);
		f[(s) * 3 + 2] += v[2] * (0.5 - r[2]);
		f[(s - strides[2]) * 3 + 2] += v[2] * (0.5 + r[2]);
	}

	template<typename TExpr>
	inline nTuple<THREE, typename Field<Geometry<Grid, 2>, TExpr>::Value>    //
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
//
//	template<int IF, typename TR> inline auto //
//	mapto(Int2Type<IF>, TR const &l, size_t s) const DECL_RET_TYPE((l[s]))
	template<int IF> inline double //
	mapto(Int2Type<IF>, double l, size_t s) const
	{
		return (l);
	}

	template<int IF> inline std::complex<double>  //
	mapto(Int2Type<IF>, std::complex<double> l, size_t s) const
	{
		return (l);
	}

	template<int IF, int N, typename TR> inline nTuple<N, TR>                //
	mapto(Int2Type<IF>, nTuple<N, TR> l, size_t s) const
	{
		return (l);
	}

	template<int IF, typename TL> inline auto //
	mapto(Int2Type<IF>, Field<Geometry<ThisType, IF>, TL> const &l,
			size_t s) const
			DECL_RET_TYPE((l[s]))

	template<typename TL> inline auto //
	mapto(Int2Type<1>, Field<Geometry<ThisType, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE( ((l[(s-s%3)/3] +l[(s-s%3)/3+strides[s%3]])*0.5) )

	template<typename TL> inline auto //
	mapto(Int2Type<2>, Field<Geometry<ThisType, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE(((
											l[(s-s%3)/3]+
											l[(s-s%3)/3+strides[(s+1)%3]]+
											l[(s-s%3)/3+strides[(s+2)%3]]+
											l[(s-s%3)/3+strides[(s+1)%3]+strides[(s+2)%3]])*0.25

							))
	template<typename TL> inline auto //
	mapto(Int2Type<3>, Field<Geometry<ThisType, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE(((
											l[(s-s%3)/3]+
											l[(s-s%3)/3+strides[(s+1)%3]]+
											l[(s-s%3)/3+strides[(s+2)%3]]+
											l[(s-s%3)/3+strides[(s+1)%3]+strides[(s+2)%3]]+
											l[(s-s%3)/3+strides[s%3]]+
											l[(s-s%3)/3+strides[s%3]+strides[(s+1)%3]]+
											l[(s-s%3)/3+strides[s%3]+strides[(s+2)%3]]+
											l[(s-s%3)/3+strides[s%3]+strides[(s+1)%3]+strides[(s+2)%3]]

									)*0.125

							))
//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

	template<int N, typename TL> inline auto //
	ExtriorDerivative(Field<Geometry<ThisType, N>, TL> const & f,
			size_t s) const
			DECL_RET_TYPE((f[s]*inv_dx[s%3]) )

	template<typename TExpr> inline auto //
	Grad(Field<Geometry<ThisType, 0>, TExpr> const & f, size_t s) const
	DECL_RET_TYPE(
			(f[(s - s % 3) / 3 + strides[s % 3]]
					- f[(s - s % 3) / 3]) * inv_dx[s % 3])

	template<typename TExpr> inline auto //
	Diverge(Field<Geometry<ThisType, 1>, TExpr> const & f, size_t s) const
	DECL_RET_TYPE(

			(f[s * 3 + 0] - f[s * 3 + 0 - 3 * strides[0]])
			* inv_dx[0] +

			(f[s * 3 + 1] - f[s * 3 + 1 - 3 * strides[1]])
			* inv_dx[1] +

			(f[s * 3 + 2] - f[s * 3 + 2 - 3 * strides[2]])
			* inv_dx[2]
	)

	template<typename TL> inline auto //
	Curl(Field<Geometry<ThisType, 1>, TL> const & f,
			size_t s) const
					DECL_RET_TYPE(
							(f[s - s %3 + (s + 2) % 3 + 3 * strides[(s + 1) % 3]] - f[s - s %3 + (s + 2) % 3])
							* inv_dx[(s + 1) % 3]
							- (f[s - s %3 + (s + 1) % 3 + 3 * strides[(s + 2) % 3]] - f[s - s %3 + (s + 1) % 3])
							* inv_dx[(s + 2) % 3]
					)

	template<typename TL> inline auto //
	Curl(Field<Geometry<ThisType, 2>, TL> const & f,
			size_t s) const
					DECL_RET_TYPE(
							(f[s - s % 3 + (s + 2) % 3]
									- f[s - s % 3 + (s + 2) % 3 - 3 * strides[(s + 1) % 3]] ) * inv_dx[(s + 1) % 3]
							-(f[s - s % 3 + (s + 1) % 3]
									- f[s - s % 3 + (s + 1) % 3 - 3 * strides[(s + 1) % 3]]) * inv_dx[(s + 2) % 3]
					)

	template<typename TExpr> inline auto //
	CurlPD(Int2Type<1>, TExpr const & expr,
			size_t s) const
					DECL_RET_TYPE( (expr.rhs_[s-s % 3 + 2 + 3 * strides[1]] - expr.rhs_[s-s % 3 + 2]) * inv_dx[1] )

	template<typename TExpr> inline auto //
	CurlPD(Int2Type<2>, TExpr const & expr,
			size_t s) const
					DECL_RET_TYPE( (-expr.rhs_[s-s % 3 + 1 + 3 * strides[2]] + expr.rhs_[s-s % 3 + 1]) * inv_dx[2])

	template<int IL, int IR, typename TL, typename TR> inline auto //
	Wedge(Field<Geometry<ThisType, IL>, TL> const &l,
			Field<Geometry<ThisType, IR>, TR> const &r,
			size_t s) const
					DECL_RET_TYPE(
							(mapto(Int2Type<IL+IR>(),l,s)*mapto(Int2Type<IL+IR>(),r,s))
					)

	template<int N, typename TL> inline auto //
	HodgeStar(Field<Geometry<ThisType, N>, TL> const & f, size_t s) const
	DECL_RET_TYPE( (mapto(Int2Type<NUM_OF_DIMS-N >(),f,s)))

	template<int N, typename TL> inline auto //
	Negate(Field<Geometry<ThisType, N>, TL> const & f, size_t s) const
	DECL_RET_TYPE( (-f[s]))

	template<int IL, typename TL, typename TR> inline auto //
	Plus(Field<Geometry<ThisType, IL>, TL> const &l,
			Field<Geometry<ThisType, IL>, TR> const &r, size_t s) const
			DECL_RET_TYPE( (l[s]+r[s]) )

	template<int IL, typename TL, typename TR> inline auto //
	Minus(Field<Geometry<ThisType, IL>, TL> const &l,
			Field<Geometry<ThisType, IL>, TR> const &r, size_t s) const
			DECL_RET_TYPE( (l[s]-r[s]) )

	template<int IL, int IR, typename TL, typename TR> inline auto //
	Multiplies(Field<Geometry<ThisType, IL>, TL> const &l,
			Field<Geometry<ThisType, IR>, TR> const &r,
			size_t s) const
					DECL_RET_TYPE( (mapto(Int2Type<IL+IR>(),l,s)*mapto(Int2Type<IL+IR>(),r,s)) )

	template<int IL, typename TL, typename TR> inline auto //
	Multiplies(Field<Geometry<ThisType, IL>, TL> const &l, TR r, size_t s) const
	DECL_RET_TYPE( (l[s]*r) )

	template<int IR, typename TL, typename TR> inline auto //
	Multiplies(TL l, Field<Geometry<ThisType, IR>, TR> const & r,
			size_t s) const
			DECL_RET_TYPE( (l*r[s]) )

	template<int IL, typename TL, typename TR> inline auto //
	Divides(Field<Geometry<ThisType, IL>, TL> const &l, TR const &r,
			size_t s) const
			DECL_RET_TYPE((l[s]/mapto(Int2Type<IL>(),r,s)))

//	template<typename TL, typename TR> inline auto //
//	Divides(Field<Geometry<ThisType, 0>, TL> const &l,
//			Field<Geometry<ThisType, 0>, TR> const &r, size_t s) const
//			DECL_RET_TYPE((l[s]/r[s]))
//
//	template<int IL, typename TL, typename TR> inline auto //
//	Divides(Field<Geometry<ThisType, IL>, TL> const &l, TR r, size_t s) const
//	DECL_RET_TYPE( (l[s]/r))
	//
//
//	template<int IPD, typename TExpr> inline auto //	Field<Geometry<Grid, 2>,
//	OpCurlPD(Int2Type<IPD>, TExpr const & expr,
//			size_t  s) const ->
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
//		typename Field<Geometry<Grid, 2>, TExpr>::Value res = 0.0;
////		if (1 == IPD)
////		{
////			res = (expr.rhs_[idx2 + 2]
////					- expr.rhs_[idx2 + 2 - 3 * strides[IPD]]) * inv_dx[IPD];
////
////		}
////		else if (2 == IPD)
////		{
////			res = (-expr.rhs_[idx2 + 1]
////					+ expr.rhs_[idx2 + 1 - 3 * strides[IPD]]) * inv_dx[IPD];
////		}
//
//		return (res);
//	}

}
;

} //namespace simpla
#endif //UNIFORM_RECT_H_

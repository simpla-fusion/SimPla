/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

#include <fetl/ntuple.h>
#include <fetl/primitives.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simpla
{
template<typename, typename > class Field;

template<typename, int> class Geometry;

template<int NDIM> struct UniformRectMesh;

/**
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh
 * */
template<>
struct UniformRectMesh<3>
{

	static const int NUM_OF_DIMS = 3;

	template<typename Element> using Container = std::vector<Element>;

	typedef size_t index_type;

	typedef nTuple<NUM_OF_DIMS, Real> coordinates_type;

	typedef std::list<index_type> chains_type;

	typedef UniformRectMesh<3> this_type;

private:
	this_type & operator=(const this_type&) = delete;

	IVec3 shift_;

	std::map<std::string, chains_type> sub_domains_;

	Real dt = 0.0;
	// Geometry
	coordinates_type xmin;
	coordinates_type xmax;
	// Topology
	nTuple<NUM_OF_DIMS, size_t> dims;
	nTuple<NUM_OF_DIMS, size_t> gw;

	nTuple<NUM_OF_DIMS, size_t> strides;
	coordinates_type inv_dx;
	coordinates_type dx;

public:

	UniformRectMesh()
	{
		Init();
	}
	~UniformRectMesh() = default;

	template<typename TCONFIG>
	void Config(TCONFIG const & vm)
	{
		vm.Get("dt", &dt);
		vm.Get("xmin", &xmin);
		vm.Get("xmax", &xmax);
		vm.Get("dims", &dims);
		vm.Get("gw", &gw);

		Init();
	}

	std::string Summary() const
	{
		std::ostringstream os;

		os

		<< "[Mesh]" << std::endl

		<< SINGLELINE << std::endl

		<< std::setw(40) << "dims = " << dims << std::endl

		<< std::setw(40) << "xmin = " << xmin << std::endl

		<< std::setw(40) << "xmax = " << xmax << std::endl

		<< std::setw(40) << "gw = " << gw << std::endl

		;
		return (os.str());
	}

	void Init()
	{
		for (int i = 0; i < NUM_OF_DIMS; ++i)
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

	}

	inline void SetExtent(coordinates_type const & pmin,
			coordinates_type const & pmax)
	{
		xmin = pmin;
		xmax = pmax;
	}

	template<typename E> inline Container<E> MakeContainer(int iform,
			E const & d = E()) const
	{
		return Container<E>(GetNumOfElements(iform), d);
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin, xmax));
	}

	template<int IFORM, typename Fun> inline
	void ForEach(Int2Type<IFORM>, Fun const &f) const
	{
		size_t num_of_ele = GetNumOfElements(IFORM);

		for (size_t s = 0; s < num_of_ele; ++s)
		{
			f(s);
		}

	}

	template<int IFORM, typename T, typename Fun> inline
	void ForEach(Int2Type<IFORM>, T & f, Fun const &fun) const
	{

		ForEach(Int2Type<IFORM>(),

		[&f,&fun] (index_type const &s)
		{
			fun(f[s]);
		}

		);

	}

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

// Property -----------------------------------------------

	inline size_t GetNumOfVertex() const
	{
		size_t res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 0) ? dims[i] : 1;
		}
		return (res);
	}
	inline size_t GetNumOfEdge() const
	{

		return (0);
	}
	inline size_t GetNumOfFace() const
	{
		return (0);
	}
	inline size_t GetNumOfCell(int iform = 0) const
	{
		size_t res = 1;
		for (int i = 0; i < 3; ++i)
		{
			res *= (dims[i] > 1) ? (dims[i] - 1) : 1;
		}
		return (res);
	}

	inline RVec3 GetCellCenter(size_t s) const
	{
		//TODO UNIMPLEMENTED!!
		RVec3 res =
		{ 0, 0, 0 };
		return (res);
	}
	inline size_t GetCellNum(IVec3 const & I) const
	{
		return (I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]);
	}
	inline size_t GetCellNum(size_t I0, size_t I1, size_t I2) const
	{
		return (I0 * strides[0] + I1 * strides[1] + I2 * strides[2]);
	}
	inline size_t GetCellNum(RVec3 x) const
	{
		IVec3 I;
		I = (x - xmin) * inv_dx;
		return ((I[0] * strides[0] + I[1] * strides[1] + I[2] * strides[2]));
	}

	inline Real GetCellVolumn(size_t s = 0) const
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

	inline Real GetCellDVolumn(size_t s = 0) const
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
	inline size_t GetNumOfElements(int iform) const
	{
		return GetNumOfComp(iform) * GetNumOfVertex();

	}

	static inline size_t GetNumOfComp(int iform)
	{
		static const int comps[4] =
		{ 1, 3, 3, 1 };

		return (comps[iform]);
	}

//	inline std::vector<size_t> Get_field_shape(int iform) const
//	{
//		int ndims = 1;
////		FIXME (iform == 1 || iform == 2) ? NDIMS + 1 : NDIMS;
//
//		std::vector<size_t> d(ndims);
//		for (int i = 0; i < NUM_OF_DIMS; ++i)
//		{
//			d[i] = dims[i];
//		}
//		if (iform == 1 || iform == 2)
//		{
//			d[NUM_OF_DIMS] = Get_numOf_comp(iform);
//		}
//		return (d);
//	}

// Coordinates transformation -------------------------------

//	nTuple<NUM_OF_DIMS, Real> CoordTransLocal2Global(index_type idx,
//			nTuple<NUM_OF_DIMS, Real> const &lcoord)
//	{
//		nTuple<NUM_OF_DIMS, Real> res;
//
//		for (int s = 0; s < 3; ++s)
//		{
//			res[s] += dx[s] * lcoord[s];
//		}
//
//		return res;
//
//	}
//
// Assign Operation --------------------------------------------
//
//	template<int IF, typename TV> TV const & //
//	GetConstValue(Field<Geometry<this_type, IF>, TV> const &f, size_t s) const
//	{
//		return (*reinterpret_cast<const TV*>(&(*f.storage)
//				+ s * f.value_size_in_bytes));
//	}
//
//	template<int IF, typename TV> TV & //
//	GetValue(Field<Geometry<this_type, IF>, TV> &f, size_t s) const
//	{
//		return (*reinterpret_cast<TV*>(&(*f.storage) + s * f.value_size_in_bytes));
//	}
//
//	template<int IFORM, typename TExpr, typename TR>
//	void Assign(Field<Geometry<this_type, IFORM>, TExpr> & lhs,
//			Field<Geometry<this_type, IFORM>, TR> const & rhs) const
//	{
//		size_t ele_num = GetNumOf_elements(IFORM);
//
//		for (size_t i = 0; i < ele_num; ++i)
//		{
//			lhs[i] = rhs[i];
//		}
//	}
//
// @NOTE the propose of this function is to assign constant vector to a field.
//   It confuses the semantics of nTuple with constant Field, and was discarded.
//	template<int IFORM, typename TExpr, int NR, typename TR>
//	void Assign(Field<Geometry<this_type, IFORM>, TExpr> & lhs, nTuple<NR, TR> rhs) const
//	{
//		ASSERT(lhs.this_type==*this);
//		Index ele_num = Get_num_of_elements(IFORM);
//
//#pragma omp parallel for
//		for (Index i = 0; i < ele_num; ++i)
//		{
//			lhs[i] = rhs[i % NR];
//		}
//	}
//	template<int IFORM, typename TL, typename TR> void //
//	Assign(Field<Geometry<this_type, IFORM>, TL>& lhs,
//			Field<Geometry<this_type, IFORM>, TR> const& rhs) const
//	{
//		ASSERT(lhs.this_type==*this);
//		{
//			std::vector<size_t> const & ele_list = Get_center_elements(IFORM);
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
//	InnerProduct(Field<Geometry<this_type, IFORM>, TLExpr> const & lhs,
//			Field<Geometry<this_type, IFORM>, TRExpr> const & rhs) const
//	{
//		typedef decltype(lhs[0] * rhs[0]) Value;
//
//		Value res;
//		res = 0;
//
//		std::vector<Index> const & ele_list = Get_center_elements(IFORM);
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
//
//	template<int IFORM, typename TL, typename TR>
//	static void //
//	Add(Field<Geometry<this_type, IFORM>, TL> & lhs,
//			Field<Geometry<this_type, IFORM>, TR> const& rhs)
//	{
//		if (lhs.grid == rhs.grid)
//		{
//			size_t size = lhs.size();
//
//			// NOTE this is parallelism of FDTD
//#pragma omp parallel for
//			for (size_t s = 0; s < size; ++s)
//			{
//				lhs[s] += rhs[s];
//			}
//		}
//
//		else
//		{
//			ERROR << "this_type mismatch!" << std::endl;
//			throw(-1);
//		}
//	}

// Mapto ----------------------------------------------------------
	/**
	 *    mapto(Int2Type<0> ,   //tarGet topology position
	 *     Field<this_type,1 , TExpr> const & vl,  //field
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
	mapto(Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l,
			size_t s) const
			DECL_RET_TYPE((l[s]))

	template<typename TL> inline auto //
	mapto(Int2Type<1>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE( ((l[(s-s%3)/3] +l[(s-s%3)/3+strides[s%3]])*0.5) )

	template<typename TL> inline auto //
	mapto(Int2Type<2>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE(((
											l[(s-s%3)/3]+
											l[(s-s%3)/3+strides[(s+1)%3]]+
											l[(s-s%3)/3+strides[(s+2)%3]]+
											l[(s-s%3)/3+strides[(s+1)%3]+strides[(s+2)%3]])*0.25

							))
	template<typename TL> inline auto //
	mapto(Int2Type<3>, Field<Geometry<this_type, 0>, TL> const &l,
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

}
;

} //namespace simpla
#include "uniform_rect_ops.h"
#endif //UNIFORM_RECT_H_

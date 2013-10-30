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
#include <deque>
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

/**
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh
 * */
struct UniformRectMesh
{

	static const int NUM_OF_DIMS = 3;

	int num_of_vertices_in_cell_ = 8;

	template<typename Element> using Container = std::vector<Element>;

	typedef size_t index_type;

	typedef nTuple<3, Real> coordinates_type;

	typedef std::list<index_type> chains_type;

	typedef UniformRectMesh this_type;

public:
	this_type & operator=(const this_type&) = delete;

	IVec3 shift_;

	std::map<std::string, chains_type> sub_domains_;

	Real dt_ = 0.0;
	// Geometry
	coordinates_type xmin_;
	coordinates_type xmax_;
	// Topology
	nTuple<NUM_OF_DIMS, size_t> dims_;
	nTuple<NUM_OF_DIMS, size_t> gw_;

	nTuple<NUM_OF_DIMS, size_t> strides_;
	coordinates_type inv_dx_;
	coordinates_type dx_;

	size_t num_of_cells_ = 0;
	size_t num_of_grid_points_ = 0;

	const size_t num_comps_per_cell_[4] =
	{ 1, 3, 3, 1 };

public:

	UniformRectMesh()
	{
		Update();
	}

	~UniformRectMesh() = default;

	template<typename TCONFIG>
	void Config(TCONFIG const & vm)
	{
		vm.Get("dt", &dt_);
		vm.Get("xmin", &xmin_);
		vm.Get("xmax", &xmax_);
		vm.Get("dims", &dims_);
		vm.Get("gw", &gw_);

		Update();
	}

	std::string Summary() const
	{
		std::ostringstream os;

		os

		<< "[Mesh]" << std::endl

		<< SINGLELINE << std::endl

		<< std::setw(40) << "dims = " << dims_ << std::endl

		<< std::setw(40) << "xmin = " << xmin_ << std::endl

		<< std::setw(40) << "xmax = " << xmax_ << std::endl

		<< std::setw(40) << "gw = " << gw_ << std::endl

		;
		return (os.str());
	}

	void Update()
	{
		num_of_cells_ = 1;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			gw_[i] = (gw_[i] * 2 > dims_[i]) ? dims_[i] / 2 : gw_[i];
			if (dims_[i] <= 1)
			{
				dims_[i] = 1;
				xmax_[i] = xmin_[i];
				dx_[i] = 0.0;
				inv_dx_[i] = 0.0;
			}
			else
			{
				dx_[i] = (xmax_[i] - xmin_[i])
						/ static_cast<Real>(dims_[i] - 1);
				inv_dx_[i] = 1.0 / dx_[i];

				num_of_cells_ *= (dims_[i] - 1);
				num_of_grid_points_ *= dims_[i];
			}
		}

		strides_[2] = 1;
		strides_[1] = dims_[2];
		strides_[0] = dims_[1] * dims_[2];

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (dims_[i] == 1)
			{
				strides_[i] = 0;
			}
		}

	}

	template<typename E> inline Container<E> MakeContainer(int iform,
			E const & d = E()) const
	{
		return std::move(Container<E>(GetNumOfGridPoints(iform), d));
	}

	template<typename Fun> inline
	void ForEach(int iform, Fun const &f) const
	{
		size_t num_of_comp = GetNumOfComp(iform);

		for (size_t i = gw_[0]; i < dims_[0] - gw_[0]; ++i)
			for (size_t j = gw_[1]; j < dims_[1] - gw_[1]; ++j)
				for (size_t k = gw_[2]; k < dims_[2] - gw_[2]; ++k)
					for (size_t m = 0; m < num_of_comp; ++m)
					{
						f(
								(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_of_comp + m);
					}

	}

	template<typename Fun> inline
	void ForEachAll(int iform, Fun const &f) const
	{
		size_t num_of_comp = GetNumOfComp(iform);

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_of_comp; ++m)
					{
						f(
								(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_of_comp + m);
					}

	}

	template<typename Fun> inline
	void ForEachBoundary(int iform, Fun const &f) const
	{
		size_t num_of_comp = GetNumOfComp(iform);

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_of_comp; ++m)
					{
						if (i >= gw_[0] && i < dims_[0] - gw_[0] &&

						j >= gw_[1] && j < dims_[1] - gw_[1] &&

						k >= gw_[2] && k < dims_[2] - gw_[2]

						)
						{
							continue;
						}
						else
						{
							f(
									(i * strides_[0] + j * strides_[1]
											+ k * strides_[2]) * num_of_comp
											+ m);
						}

					}

	}

	void MakeCycleMap(int iform, std::map<index_type, index_type> &ma,
			unsigned int flag = 7) const
	{
		size_t num_of_comp = GetNumOfComp(iform);

		nTuple<NUM_OF_DIMS, size_t> L =
		{ dims_[0] - 2 * gw_[0], dims_[1] - 2 * gw_[1], dims_[2] - 2 * gw_[2] };

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{

					index_type s = i * strides_[0] + j * strides_[1]
							+ k * strides_[2];
					index_type t = s;

					if (flag & 1)
					{
						if (i < gw_[0])
						{
							t += L[0] * strides_[0];
						}
						else if (i >= dims_[0] - gw_[0])
						{
							t -= L[0] * strides_[0];
						}
					}

					if (flag & 2)
					{
						if (j < gw_[1])
						{
							t += L[1] * strides_[1];
						}
						else if (j >= dims_[1] - gw_[1])
						{
							t -= L[1] * strides_[1];
						}
					}

					if (flag & 4)
					{
						if (k < gw_[2])
						{
							t += L[2] * strides_[2];
						}
						else if (k >= dims_[2] - gw_[2])
						{
							t -= L[2] * strides_[2];
						}
					}
					if (s != t)
					{
						for (size_t m = 0; m < num_of_comp; ++m)
						{
							ma[s * num_of_comp + m] = t * num_of_comp + m;
						}
					}

				}
	}

	template<int IFORM, typename T1>
	void Print(Field<Geometry<this_type, IFORM>, T1> const & f) const
	{
		size_t num_of_comp = GetNumOfComp(IFORM);

		for (size_t i = 0; i < dims_[0]; ++i)
		{
			std::cout << "--------------------------------------------------"
					<< std::endl;
			for (size_t j = 0; j < dims_[1]; ++j)
			{
				std::cout << std::endl;
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					std::cout << "(";
					for (size_t m = 0; m < num_of_comp; ++m)
					{
						std::cout
								<< f[(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_of_comp + m]
								<< " ";
					}
					std::cout << ") ";
				}
				std::cout << std::endl;
			}

		}
		std::cout << std::endl;

	}
	template<int IFORM, typename T1, typename T2>
	void UpdateBoundary(std::map<index_type, index_type> const & m,
			Field<Geometry<this_type, IFORM>, T1> & src,
			Field<Geometry<this_type, IFORM>, T2> & dest) const
	{
		for (auto & p : m)
		{
			dest[p.first] = src[p.second];
		}

	}

	template<int IFORM, typename T1>
	void UpdateCyCleBoundary(Field<Geometry<this_type, IFORM>, T1> & f) const
	{
		std::map<index_type, index_type> m;
		MakeCycleMap(IFORM, m);
		UpdateBoundary(m, f, f);
	}

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

// Property -----------------------------------------------

	inline size_t GetNumOfComp(int iform) const
	{
		return (num_comps_per_cell_[iform]);
	}

	inline size_t GetNumOfGridPoints(int iform) const
	{

		return (num_of_grid_points_ * num_comps_per_cell_[iform]);
	}

	Real GetDt() const
	{
		return dt_;
	}

	void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}

	size_t GetNumCompsPerCell(int iform) const
	{
		return num_comps_per_cell_[iform];
	}

	size_t GetNumOfCells() const
	{
		return num_of_cells_;
	}

	inline void SetExtent(coordinates_type const & pmin,
			coordinates_type const & pmax)
	{
		xmin_ = pmin;
		xmax_ = pmax;

		Update();
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline void SetDimension(nTuple<NUM_OF_DIMS, size_t> const & pdims)
	{
		dims_ = pdims;

		Update();
	}
	inline nTuple<NUM_OF_DIMS, size_t> const & GetDimension() const
	{
		return dims_;
	}

	inline void GetVerticesOfCell(index_type idx, index_type points[]) const
	{
		// 0 0 0
		points[0] = idx;
		// 0 1 0
		points[1] = idx + strides_[0];
		// 0 0 1
		points[2] = idx + strides_[1];
		// 0 1 1
		points[3] = idx + strides_[0] + strides_[1];
		// 1 0 0
		points[4] = idx + strides_[2];
		// 1 1 0
		points[5] = idx + strides_[0] + strides_[2];
		// 1 0 1
		points[6] = idx + strides_[1] + strides_[2];
		// 1 1 1
		points[7] = idx + strides_[0] + strides_[1] + strides_[2];

	}

	inline bool SearchCell(index_type hint, coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const
	{

		size_t idx = 0;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			idx += static_cast<size_t>(std::modf((x[i] - xmin_[i]) * inv_dx_[i],
					&((*pcoords)[i]))) * strides_[i];
		}

		return idx == hint;
	}

	/**
	 * Locate the cell containing a specified point.
	 * @param x
	 * @param pcoords local parameter coordinates
	 * @return index of cell
	 */
	inline index_type SearchCell(coordinates_type const &x,
			coordinates_type *pcoords) const
	{

		size_t idx = 0;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			idx += static_cast<size_t>(std::modf((x[i] - xmin_[i]) * inv_dx_[i],
					&((*pcoords)[i]))) * strides_[i];
		}

		return idx;
	}

	inline void CalcuateWeight(coordinates_type const &pcoords,
			Real *weight) const
	{
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];

		weight[0] = (1.0 - r) * (1.0 - s) * (1.0 - t);
		weight[1] = r * (1.0 - s) * (1.0 - t);
		weight[2] = (1.0 - r) * s * (1.0 - t);
		weight[3] = r * s * (1.0 - t);
		weight[4] = (1.0 - r) * (1.0 - s) * t;
		weight[5] = r * (1.0 - s) * t;
		weight[6] = (1.0 - r) * s * t;
		weight[7] = r * s * t;
	}

	template<typename TF>
	inline typename Field<Geometry<this_type, 0>, TF>::value_type Gather(
			Field<Geometry<this_type, 0>, TF> const &f,
			coordinates_type const &x) const
	{
		index_type points[num_of_vertices_in_cell_];
		coordinates_type pcoords;
		Real weight[num_of_vertices_in_cell_];
		GetVerticesOfCell(SearchCell(x, &pcoords), points);
		CalcuateWeight(pcoords, weight);

		return std::move(Gather(f, points, weight));

	}

	template<typename TF>
	inline typename TF::value_type Gather(
			Field<Geometry<this_type, 0>, TF> const &f,
			index_type const points[], Real const weight[]) const
	{
		typename TF::value_type res;

		res = 0;

		for (int i = 0; i < num_of_vertices_in_cell_; ++i)
		{
			res += f[points[i]] * weight[i];
		}
		return std::move(res);
	}

	template<typename TV, typename TF>
	inline void Scatter(TV const & v, coordinates_type const &x,
			Field<Geometry<this_type, 0>, TF> &f) const
	{
		index_type points[num_of_vertices_in_cell_];
		Real weight[num_of_vertices_in_cell_];
		coordinates_type pcoords;

		GetVerticesOfCell(SearchCell(x, &pcoords), points);
		CalcuateWeight(pcoords, weight);

		Scatter(v, points, weight, f);

	}

	template<typename TV, typename TF>
	inline void Scatter(TV const &v, index_type const points[],
			Real const weight[], Field<Geometry<this_type, 0>, TF> &f) const
	{
		for (int i = 0; i < num_of_vertices_in_cell_; ++i)
		{
			f[points[i]] += v * weight[i];
		}
	}

	inline Real GetCellVolumn(index_type s = 0) const
	{
		Real res = 1.0;
		for (int i = 0; i < 3; ++i)
		{
			if (!isinf(dx_[i]))
			{
				res *= (dims_[i] - 1) * dx_[i];
			}
		}

		return (res);
	}

	inline Real GetCellDVolumn(size_t s = 0) const
	{
		Real res = 1.0;
		for (int i = 0; i < 3; ++i)
		{
			if (!isinf(dx_[i]) && dx_[i] > 0)
			{
				res *= dx_[i];
			}
		}

		return (res);
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

	template<int IF, int N, typename TR> inline nTuple<N, TR>            //
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
					DECL_RET_TYPE( ((l[(s-s%3)/3] +l[(s-s%3)/3+strides_[s%3]])*0.5) )

	template<typename TL> inline auto //
	mapto(Int2Type<2>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE(((
											l[(s-s%3)/3]+
											l[(s-s%3)/3+strides_[(s+1)%3]]+
											l[(s-s%3)/3+strides_[(s+2)%3]]+
											l[(s-s%3)/3+strides_[(s+1)%3]+strides_[(s+2)%3]])*0.25

							))
	template<typename TL> inline auto //
	mapto(Int2Type<3>, Field<Geometry<this_type, 0>, TL> const &l,
			size_t s) const
					DECL_RET_TYPE(((
											l[(s-s%3)/3]+
											l[(s-s%3)/3+strides_[(s+1)%3]]+
											l[(s-s%3)/3+strides_[(s+2)%3]]+
											l[(s-s%3)/3+strides_[(s+1)%3]+strides_[(s+2)%3]]+
											l[(s-s%3)/3+strides_[s%3]]+
											l[(s-s%3)/3+strides_[s%3]+strides_[(s+1)%3]]+
											l[(s-s%3)/3+strides_[s%3]+strides_[(s+2)%3]]+
											l[(s-s%3)/3+strides_[s%3]+strides_[(s+1)%3]+strides_[(s+2)%3]]

									)*0.125

							))

}
;

} //namespace simpla
#include "uniform_rect_ops.h"
#endif //UNIFORM_RECT_H_

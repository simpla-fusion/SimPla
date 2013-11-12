/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * fetl/grid/uniform_rect.h
 *
 * Created on: 2009-4-19
 * Author: salmon
 */
#ifndef UNIFORM_RECT_H_
#define UNIFORM_RECT_H_

#include <fetl/field.h>
#include <fetl/geometry.h>
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
#include <type_traits>
#include <utility>
#include <vector>
#include "utilities/log.h"

namespace simpla
{

/**
 *
 *  @brief UniformRectMesh -- Uniform rectangular structured grid.
 *  @ingroup mesh
 * */
struct UniformRectMesh
{
	typedef UniformRectMesh this_type;

	static const int NUM_OF_DIMS = 3;

	template<typename Element> struct Container
	{
		typedef std::vector<Element> type;
	};

	typedef size_t index_type;

	typedef nTuple<3, Real> coordinates_type;

	typedef std::list<index_type> chains_type;

	this_type & operator=(const this_type&) = delete;

	nTuple<NUM_OF_DIMS, size_t> shift_;

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

	static const int num_vertices_in_cell_ = 8;

	size_t num_cells_ = 0;
	size_t num_grid_points_ = 0;

	Real cell_volume_ = 1.0;
	Real d_cell_volume_ = 1.0;

	const size_t num_comps_per_cell_[4] =
	{ 1, 3, 3, 1 };

	UniformRectMesh()
	{
		Update();
	}

	~UniformRectMesh() = default;

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

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

	void Update()
	{
		num_cells_ = 1;
		num_grid_points_ = 1;
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

				num_cells_ *= (dims_[i] - 1);
				num_grid_points_ *= dims_[i];
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

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (!isinf(dx_[i]))
			{
				cell_volume_ *= (dims_[i] - 1) * dx_[i];
			}
		}

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (!isinf(dx_[i]) && dx_[i] > 0)
			{
				d_cell_volume_ *= dx_[i];
			}
		}

	}

	template<typename E> inline typename Container<E>::type MakeContainer(
			int iform, E const & d = E()) const
	{
		return std::move(
				typename Container<E>::type(GetNumOfGridPoints(iform), d));
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

	template<int IFORM, typename T1>
	void Print(Field<Geometry<this_type, IFORM>, T1> const & f) const
	{
		size_t num_comp = num_comps_per_cell_[IFORM];

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
					for (size_t m = 0; m < num_comp; ++m)
					{
						std::cout
								<< f[(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_comp + m]
								<< " ";
					}
					std::cout << ") ";
				}
				std::cout << std::endl;
			}

		}
		std::cout << std::endl;

	}

	template<typename Fun, typename ... Args> inline
	void ForAll(int iform, Fun const &f, Args & ... args) const
	{
		size_t num_comp = num_comps_per_cell_[iform];

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_comp; ++m)
					{
						f(
								(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_comp + m,
								std::forward<Args>(args)...);
					}

	}

	template<typename Fun, typename ... Args> inline
	void ForAllCell(Fun const &fun, Args &... args) const
	{

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
				{
					size_t s = (i * strides_[0] + j * strides_[1]
							+ k * strides_[2]);

					fun(s, args...);
				}

	}

	template<typename Fun> inline
	void ForEach(int iform, Fun const &f) const
	{
		size_t num_comp = num_comps_per_cell_[iform];

		for (size_t i = gw_[0]; i < dims_[0] - gw_[0]; ++i)
			for (size_t j = gw_[1]; j < dims_[1] - gw_[1]; ++j)
				for (size_t k = gw_[2]; k < dims_[2] - gw_[2]; ++k)
					for (size_t m = 0; m < num_comp; ++m)
					{
						f(
								(i * strides_[0] + j * strides_[1]
										+ k * strides_[2]) * num_comp + m);
					}

	}

	template<typename Fun> inline
	void ForEachBoundary(int iform, Fun const &f) const
	{
		size_t num_comp = num_comps_per_cell_[iform];

		for (size_t i = 0; i < dims_[0]; ++i)
			for (size_t j = 0; j < dims_[1]; ++j)
				for (size_t k = 0; k < dims_[2]; ++k)
					for (size_t m = 0; m < num_comp; ++m)
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
											+ k * strides_[2]) * num_comp + m);
						}

					}

	}

	void MakeCycleMap(int iform, std::map<index_type, index_type> &ma,
			unsigned int flag = 7) const
	{
		size_t num_comp = num_comps_per_cell_[iform];

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
						for (size_t m = 0; m < num_comp; ++m)
						{
							ma[s * num_comp + m] = t * num_comp + m;
						}
					}

				}
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

	// Properties of UniformRectMesh --------------------------------------
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
	// General Property -----------------------------------------------

	inline Real GetDt() const
	{
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}

	inline size_t GetNumOfGridPoints(int iform) const
	{

		return (num_grid_points_ * num_comps_per_cell_[iform]);
	}

	inline Real GetCellVolume() const
	{
		return cell_volume_;
	}

	inline Real GetDCellVolume() const
	{
		return d_cell_volume_;
	}

	/**
	 * Locate the cell containing a specified point.
	 * @param x
	 * @param pcoords local parameter coordinates
	 * @return index of cell
	 */
	inline index_type SearchCell(coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const
	{

		size_t idx = 0;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			double e;

			idx += static_cast<size_t>(std::modf((x[i] - xmin_[i]) * inv_dx_[i],
					&e)) * strides_[i];

			if (pcoords != nullptr)
				(*pcoords)[i] = e;
		}

		return idx;
	}
	/**
	 *  Speed up version SearchCell
	 * @param
	 * @param x
	 * @param pcoords
	 * @return
	 */
	inline index_type SearchCell(index_type const &hint_idx,
			coordinates_type const &x,
			coordinates_type *pcoords = nullptr) const
	{
		return SearchCell(x, pcoords);
	}

	inline std::vector<coordinates_type> GetCellShape(index_type s) const
	{
		std::vector<coordinates_type> res;
		coordinates_type r0 =
		{ 0, 0, 0 };
		res.push_back(GetGlobalCoordinates(s, r0));
		coordinates_type r1 =
		{ 1, 1, 1 };
		res.push_back(GetGlobalCoordinates(s, r1));

		return res;
	}

	inline coordinates_type GetGlobalCoordinates(index_type s,
			coordinates_type const &r) const
	{
		coordinates_type res;

		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			if (strides_[i] != 0)
			{
				res[i] = (s / strides_[i]) * dx_[i] + xmin_[i];

				s %= strides_[i];
			}
			else
			{
				res[i] = xmin_[i];
			}
		}
		res += r * dx_;

		return res;

	}
	template<typename PList>
	inline void GetAffectedPoints(Int2Type<0>, index_type const & idx,
			PList & points, int affect_region = 1) const
	{

		points.resize(8);
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

	template<typename PList>
	inline void GetAffectedPoints(Int2Type<1>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}

	template<typename PList>
	inline void GetAffectedPoints(Int2Type<2>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}
	template<typename PList>
	inline void GetAffectedPoints(Int2Type<3>, index_type const & idx,
			PList& points, int affect_region = 1) const
	{

		// 0 0 0

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<0>, coordinates_type const &pcoords,
			TW & weights, int affect_region = 1) const
	{
		weights.resize(8);
		Real r = (pcoords)[0], s = (pcoords)[1], t = (pcoords)[2];

		weights[0] = (1.0 - r) * (1.0 - s) * (1.0 - t);
		weights[1] = r * (1.0 - s) * (1.0 - t);
		weights[2] = (1.0 - r) * s * (1.0 - t);
		weights[3] = r * s * (1.0 - t);
		weights[4] = (1.0 - r) * (1.0 - s) * t;
		weights[5] = r * (1.0 - s) * t;
		weights[6] = (1.0 - r) * s * t;
		weights[7] = r * s * t;
	}
	template<typename TW>
	inline void CalcuateWeights(Int2Type<1>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<2>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

	template<typename TW>
	inline void CalcuateWeights(Int2Type<3>, coordinates_type const &pcoords,
			TW & weight, int affect_region = 1) const
	{

	}

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

	template<int IF, typename TL>
	inline auto //
	mapto(Int2Type<IF>, Field<Geometry<this_type, IF>, TL> const &l,
			size_t s) const
			DECL_RET_TYPE ((l[s]))

	template<typename TL>
	inline auto //
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

	template<typename TF>
	auto Index(TF const & f, index_type const & s) const->decltype(f[s])
	{
		return f[s];
	}

	template<typename T> inline typename std::enable_if<
			is_arithmetic_scalar<T>::value, T>::type Index(T const & f,
			index_type const &) const
	{
		return f;
	}

}
;

namespace UniformRectMeshDefine
{
typedef UniformRectMesh Mesh;

template<int IFORM, typename T> using Form = Field<Geometry<Mesh,IFORM>,T >;

typedef Form<0, Real> ZeroForm;
typedef Form<1, Real> OneForm;
typedef Form<2, Real> TwoForm;
typedef Form<3, Real> ThreeForm;

typedef Form<0, nTuple<3, Real> > VecZeroForm;
typedef Form<1, nTuple<3, Real> > VecOneForm;
typedef Form<2, nTuple<3, Real> > VecTwoForm;
typedef Form<3, nTuple<3, Real> > VecThreeForm;

typedef Form<0, Real> ScalarField;
typedef Form<0, nTuple<3, Real> > VecField;

typedef Form<0, Real> RZeroForm;
typedef Form<1, Real> ROneForm;
typedef Form<2, Real> RTwoForm;
typedef Form<3, Real> RThreeForm;

typedef Form<0, nTuple<3, Real> > RVecZeroForm;
typedef Form<1, nTuple<3, Real> > RVecOneForm;
typedef Form<2, nTuple<3, Real> > RVecTwoForm;
typedef Form<3, nTuple<3, Real> > RVecThreeForm;

typedef Form<0, Real> RScalarField;
typedef Form<0, nTuple<3, Real> > RVecField;

typedef Form<0, Complex> CZeroForm;
typedef Form<1, Complex> COneForm;
typedef Form<2, Complex> CTwoForm;
typedef Form<3, Complex> CThreeForm;

typedef Form<0, nTuple<3, Complex> > CVecZeroForm;
typedef Form<0, nTuple<3, Complex> > CVecOneForm;
typedef Form<0, nTuple<3, Complex> > CVecTwoForm;
typedef Form<3, nTuple<3, Complex> > CVecThreeForm;

typedef Form<0, Complex> CScalarField;
typedef Form<0, nTuple<3, Complex> > CVecField;

}  // namespace UniformRectMeshDefine

//	inline std::vector<size_t> Get_field_shape(int iform) const
//	{
//		int ndims = 1;
// //		FIXME (iform == 1 || iform == 2) ? NDIMS + 1 : NDIMS;
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
//		Index ele_num = Get_num_elements(IFORM);
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

//-----------------------------------------
// Vector Arithmetic
//-----------------------------------------

template<int N, typename TL, typename TI> inline auto _OpEval(
		Int2Type<EXTRIORDERIVATIVE>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f, TI s)
		DECL_RET_TYPE((f[s]*f.mesh.inv_dx_[s%3]))

template<typename TExpr, typename TI> inline auto _OpEval(Int2Type<GRAD>,
		Field<Geometry<UniformRectMesh, 0>, TExpr> const & f,
		TI s)
				DECL_RET_TYPE(
						(f[(s - s % 3) / 3 + f.mesh.strides_[s % 3]] - f[(s - s % 3) / 3]) * f.mesh.inv_dx_[s % 3])

template<typename TExpr, typename TI> inline auto _OpEval(Int2Type<DIVERGE>,
		Field<Geometry<UniformRectMesh, 1>, TExpr> const & f,
		TI s)
				DECL_RET_TYPE(

						(f[s * 3 + 0] - f[s * 3 + 0 - 3 * f.mesh.strides_[0]]) * f.mesh.inv_dx_[0] +

						(f[s * 3 + 1] - f[s * 3 + 1 - 3 * f.mesh.strides_[1]]) * f.mesh.inv_dx_[1] +

						(f[s * 3 + 2] - f[s * 3 + 2 - 3 * f.mesh.strides_[2]]) * f.mesh.inv_dx_[2])

template<typename TL, typename TI> inline auto _OpEval(Int2Type<CURL>,
		Field<Geometry<UniformRectMesh, 1>, TL> const & f,
		TI s)
				DECL_RET_TYPE(
						(f[s - s %3 + (s + 2) % 3 + 3 * f.mesh.strides_[(s + 1) % 3]] - f[s - s %3 + (s + 2) % 3]) * f.mesh.inv_dx_[(s + 1) % 3]

						- (f[s - s %3 + (s + 1) % 3 + 3 * f.mesh.strides_[(s + 2) % 3]] - f[s - s %3 + (s + 1) % 3]) * f.mesh.inv_dx_[(s + 2) % 3]

				)

template<typename TL, typename TI> inline auto _OpEval(Int2Type<CURL>,
		Field<Geometry<UniformRectMesh, 2>, TL> const & f,
		TI s)
				DECL_RET_TYPE(
						(f[s - s % 3 + (s + 2) % 3] - f[s - s % 3 + (s + 2) % 3 - 3 * f.mesh.strides_[(s + 1) % 3]] ) * f.mesh.inv_dx_[(s + 1) % 3]

						-(f[s - s % 3 + (s + 1) % 3] - f[s - s % 3 + (s + 1) % 3 - 3 * f.mesh.strides_[(s + 2) % 3]]) * f.mesh.inv_dx_[(s + 2) % 3]

				)

//template<typename TL,typename TI>  inline auto //
//_OpEval(<CURLPD1>, Field<Geometry<UniformRectMesh, 1>, TL> const & f,
//		TI s)
//				DECL_RET_TYPE(
//						(f.rhs_[s-s % 3 + 2 + 3 * f.mesh.strides_[1]] - f.rhs_[s-s % 3 + 2]) * f.mesh.inv_dx_[1])
//
//template<typename TL,typename TI>  inline auto //
//_OpEval(<CURLPD2>, Field<Geometry<UniformRectMesh, 2>, TL> const & f,
//		TI s)
//				DECL_RET_TYPE(
//						(-f.rhs_[s-s % 3 + 1 + 3 * f.mesh.strides_[2]] + f.rhs_[s-s % 3 + 1]) * f.mesh.inv_dx_[2])

template<int IL, int IR, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<WEDGE>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		TI s)
				DECL_RET_TYPE(
						(l.mesh.mapto(Int2Type<IL+IR>(),l,s)*r.mesh.mapto(Int2Type<IL+IR>(),r,s)))

template<int N, typename TL, typename TI> inline auto _OpEval(
		Int2Type<HODGESTAR>, Field<Geometry<UniformRectMesh, N>, TL> const & f,
		TI s)
				DECL_RET_TYPE((f.mesh.mapto(Int2Type<UniformRectMesh::NUM_OF_DIMS-N >(),f,s)))

template<int N, typename TL, typename TI> inline auto _OpEval(Int2Type<NEGATE>,
		Field<Geometry<UniformRectMesh, N>, TL> const & f, TI s)
		DECL_RET_TYPE((-f[s]))

template<int IL, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<PLUS>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r, TI s)
		DECL_RET_TYPE((l[s]+r[s]))

template<int IL, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<MINUS>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IL>, TR> const &r, TI s)
		DECL_RET_TYPE((l[s]-r[s]))

template<int IL, int IR, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		Field<Geometry<UniformRectMesh, IR>, TR> const &r,
		TI s)
				DECL_RET_TYPE( (l.mesh.mapto(Int2Type<IL+IR>(),l,s)*r.mesh.mapto(Int2Type<IL+IR>(),r,s)) )

template<int IL, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, TI s)
		DECL_RET_TYPE((l[s] * r))

template<int IR, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<MULTIPLIES>, TL const & l,
		Field<Geometry<UniformRectMesh, IR>, TR> const & r, TI s)
		DECL_RET_TYPE((l * r[s]))

template<int IL, typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<DIVIDES>, Field<Geometry<UniformRectMesh, IL>, TL> const &l,
		TR const &r, TI s)
		DECL_RET_TYPE((l[s]/l.mesh.mapto(Int2Type<IL>(),r,s)))

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<DOT>, Field<Geometry<UniformRectMesh, 0>, TL> const &l,
		Field<Geometry<UniformRectMesh, 0>, TR> const &r, TI s)
		DECL_RET_TYPE((Dot(l[s],r[s])) )

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<DOT>, Field<Geometry<UniformRectMesh, 0>, TL> const &l,
		nTuple<3, TR> const &r, TI s)
		DECL_RET_TYPE((Dot(l[s] , r)))

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<DOT>, nTuple<3, TL> const & l,
		Field<Geometry<UniformRectMesh, 0>, TR> const & r, TI s)
		DECL_RET_TYPE((Dot(l , r[s])))

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<CROSS>, Field<Geometry<UniformRectMesh, 0>, TL> const &l,
		Field<Geometry<UniformRectMesh, 0>, TR> const &r, TI s)
		DECL_RET_TYPE( (Cross(l[s],r[s])))

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<CROSS>, Field<Geometry<UniformRectMesh, 0>, TL> const &l,
		nTuple<3, TR> const &r, TI s)
		DECL_RET_TYPE((Cross(l[s] , r)))

template<typename TL, typename TR, typename TI> inline auto _OpEval(
		Int2Type<CROSS>, nTuple<3, TL> const & l,
		Field<Geometry<UniformRectMesh, 0>, TR> const & r, TI s)
		DECL_RET_TYPE((Cross(l , r[s])))

//template
//	Divides(Field<Geometry<ThisType, 0>, TL> const &l,
//			Field<Geometry<ThisType, 0>, TR> const &r, TI  s)
//			DECL_RET_TYPE((l[s]/r[s]))
//
//	template<int IL, typename TL, typename TR,typename TI>  inline auto //
//	Divides(Field<Geometry<ThisType, IL>, TL> const &l, TR r, TI  s)
//	DECL_RET_TYPE( (l[s]/r))
//
//
//	template<int IPD, typename TExpr,typename TI>  inline auto //	Field<Geometry<this_type, 2>,
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
////					- expr.rhs_[idx2 + 2 - 3 * strides_[IPD]]) * f.mesh.inv_dx_[IPD];
////
////		}
////		else if (2 == IPD)
////		{
////			res = (-expr.rhs_[idx2 + 1]
////					+ expr.rhs_[idx2 + 1 - 3 * strides_[IPD]]) * f.mesh.inv_dx_[IPD];
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

#endif //UNIFORM_RECT_H_

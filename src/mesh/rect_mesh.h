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
#include "../utilities/utilities.h"
#include "../utilities/memory_pool.h"
#include "../physics/physical_constants.h"
#include "octree_forest.h"
namespace simpla
{

class RectMesh: public OcForest
{
	typedef OcForest base_type;

	typedef RectMesh this_type;

public:

	RectMesh();
	~RectMesh();

	this_type & operator=(const this_type&) = delete;

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************

	static inline std::string GetTypeName()
	{
		return "RectMesh";
	}

	inline std::string GetTopologyTypeAsString() const
	{
		return ToString(GetRealNumDimension()) + "DRectMesh";
	}

	//***************************************************************************************************
	//* Media Tags
	//***************************************************************************************************
	//
	//private:
	//
	//	MediaTag<this_type> tags_;
	//public:
	//
	//	typedef typename MediaTag<this_type>::tag_type tag_type;
	//	MediaTag<this_type> & tags()
	//	{
	//		return tags_;
	//	}
	//	MediaTag<this_type> const& tags() const
	//	{
	//
	//		return tags_;
	//	}
	//
	//	//***************************************************************************************************
	//	//* Configure
	//	//***************************************************************************************************
	//
	//	template<typename TDict> void Load(TDict const &cfg);
	//
	//	std::ostream & Save(std::ostream &vm) const;
	//
	//	void Update();
	//
	//	inline bool operator==(this_type const & r) const
	//	{
	//		return (this == &r);
	//	}
	//
	//
	//
	//	//***************************************************************************************************
	//	//* Time
	//	//***************************************************************************************************
	//
	//private:
	//	Real dt_ = 0.0; //!< time step
	//	Real time_ = 0.0;
	//public:
	//
	//	void NextTimeStep()
	//	{
	//		time_ += dt_;
	//	}
	//	Real GetTime() const
	//	{
	//		return time_;
	//	}
	//
	//	void GetTime(Real t)
	//	{
	//		time_ = t;
	//	}
	//	inline Real GetDt() const
	//	{
	//		CheckCourant();
	//		return dt_;
	//	}
	//
	//	inline void SetDt(Real dt = 0.0)
	//	{
	//		dt_ = dt;
	//		Update();
	//	}
	//	double CheckCourant() const
	//	{
	//		DEFINE_GLOBAL_PHYSICAL_CONST
	//
	//		nTuple<3, Real> inv_dx_;
	//		inv_dx_ = 1.0 / GetDx() / (xmax_ - xmin_);
	//
	//		Real res = 0.0;
	//
	//		for (int i = 0; i < 3; ++i)
	//			res += inv_dx_[i] * inv_dx_[i];
	//
	//		return std::sqrt(res) * speed_of_light * dt_;
	//	}
	//
	//	void FixCourant(Real a)
	//	{
	//		dt_ *= a / CheckCourant();
	//	}
	//
	//	//***************************************************************************************************
	//	//* Container: storage depend
	//	//***************************************************************************************************
	//
	//	template<typename TV> using Container=std::shared_ptr<TV>;
	//
	//	nTuple<NUM_OF_DIMS, size_type> strides_ = { 0, 0, 0 };
	//
	//	inline nTuple<NUM_OF_DIMS, index_type> const & GetStrides() const
	//	{
	//		return strides_;
	//	}
	//
	//	inline index_type GetNumOfElements(int iform) const
	//	{
	//
	//		return (num_grid_points_ * num_comps_per_cell_[iform]);
	//	}
	//
	//	template<int iform, typename TV> inline Container<TV> MakeContainer() const
	//	{
	//		return (MEMPOOL.allocate_shared_ptr < TV > (GetNumOfElements(iform)));
	//	}

	//***************************************************************************************************
	// Geometric properties
	//***************************************************************************************************

	typedef nTuple<3, Real> coordinates_type;

	coordinates_type xmin_ = { 0, 0, 0 };

	coordinates_type xmax_ = { 10, 10, 10 };

	coordinates_type inv_L = { 1.0, 1.0, 1.0 };

	template<int IN, typename T>
	inline void SetExtent(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NUM_OF_DIMS ? IN : NUM_OF_DIMS;

		for (int i = 0; i < n; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];
		}

		for (int i = n; i < NUM_OF_DIMS; ++i)
		{
			xmin_[i] = 0;
			xmax_[i] = 0;
		}
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	/**
	 * Mertic
	 * @param s coodinates
	 * @param m suffix
	 * @param n suffix
	 * @return
	 */
	inline Real g(index_type const & s, int m = 0, int n = 0) const
	{
		return 1.0;
	}

	//***************************************************************************************************
	//* Mesh operation

	inline coordinates_type GetCoordinates(index_type const &s) const
	{
		coordinates_type res = base_type::GetCoordinates(s);
		res = xmin_ + (xmax_ - xmin_) * base_type::GetCoordinates(s);
		return std::move(res);
	}
//
//	template<typename TV>
//	TV & get_value(Container<TV> & d, index_type s) const
//	{
//		return *(d.get() + HashRootIndex(s, strides_));
//	}
//	template<typename TV>
//	TV const & get_value(Container<TV> const& d, index_type s) const
//	{
//		return *(d.get() + HashRootIndex(s, strides_));
//	}
	//***************************************************************************************************
	// Metric
	//***************************************************************************************************

	/**
	 *  Mapto -
	 *    mapto(Int2Type<VERTEX> ,   //tarGet topology position
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

	template<int IF, typename T> inline auto	//
	mapto(Int2Type<IF>, T const &l, ...) const
	ENABLE_IF_DECL_RET_TYPE(is_primitive<T>::value,l)

	template<int IF, typename TL> inline typename Field<Geometry<this_type, IF>, TL>::value_type mapto(Int2Type<IF>,
	        Field<Geometry<this_type, IF>, TL> const &l, index_type s) const
	{
		return g_[s];
	}
	template<int IF, int IR, typename TR> inline typename Field<Geometry<this_type, IR>, TR>::value_type mapto(
	        Int2Type<IF>, Field<Geometry<this_type, IR>, TR> const &l, index_type s) const
	{
		index_type v[MAX_NUM_VERTEX_PER_CEL];

		typename Field<Geometry<this_type, IF>, TL>::value_type res;

		res = 0;

		int n = GetAdjacentCells(Int2Type<IR>(), Int2Type<IF>(), s, v);

		for (int i = 0; i < n; ++i)
		{
			res += l[i];
		}
		return res / static_cast<Real>(n);
	}

	Real idh(index_type s, int N) const
	{
		return static_cast<Real>(1UL << (INDEX_DIGITS - index_digits_[N]));
	}

	// for Orthogonal coordinates
	Real g_(index_type s, unsigned int a = 1) const
	{
		return 1.0;
	}
	//-----------------------------------------
	// Vector Arithmetic
	//-----------------------------------------
	template<typename TExpr> inline typename Field<Geometry<this_type, VERTEX>, TExpr>::value_type OpEval(
	        Int2Type<GRAD>, Field<Geometry<this_type, VERTEX>, TExpr> const & f, index_type s) const
	{
		auto d = s & (_MA >> (s.H + 1));
		return ((f[s + d] - f[s - d]) * inv_L[0] * idh(s, _N(s)) / g_(s, 1L << _N(s)));
	}

	template<typename TExpr> inline typename Field<Geometry<this_type, EDGE>, TExpr>::value_type OpEval(
	        Int2Type<DIVERGE>, Field<Geometry<this_type, EDGE>, TExpr> const & f, index_type s) const
	{
		auto d1 = (_MI >> (s.H + 1));
		auto d2 = (_MJ >> (s.H + 1));
		auto d3 = (_MK >> (s.H + 1));

		return (

		(f[s + d1] * g_(s + d1, 2 + 4) - f[s - d1] * g_(s - d1, 2 + 4)) * inv_L[0] * idh(s, 0) +

		(f[s + d2] * g_(s + d2, 4 + 1) - f[s - d2] * g_(s - d2, 4 + 1)) * inv_L[1] * idh(s, 1) +

		(f[s + d3] * g_(s + d3, 1 + 2) - f[s - d3] * g_(s - d3, 1 + 2)) * inv_L[2] * idh(s, 2)

		) / g_(s, ~0U);
	}

	template<typename TExpr> inline typename Field<Geometry<this_type, EDGE>, TExpr>::value_type OpEval(Int2Type<CURL>,
	        Field<Geometry<this_type, EDGE>, TExpr> const & f, index_type s) const
	{
		auto d3 = (_RR(s) >> (s.H + 1));
		auto d2 = (_R(s) >> (s.H + 1));

		unsigned int n1 = _N(s);
		unsigned int n2 = (n1 + 1) % 3;
		unsigned int n3 = (n2 + 1) % 3;
		return (

		(f[s + d2] * g_(s + d2, 1L << n2) - f[s - d2] * g_(s - d2, 1L << n2)) * inv_L[n2] * idh(s, n2) -

		(f[s + d3] * g_(s + d3, 1L << n3) - f[s - d3] * g_(s - d3, 1L << n3)) * inv_L[n3] * idh(s, n3)

		) / g_(s, ~(1L << n1));
	}

	template<typename TL> inline typename Field<Geometry<this_type, FACE>, TL>::value_type OpEval(Int2Type<CURL>,
	        Field<Geometry<this_type, FACE>, TL> const & f, index_type s) const
	{
		auto d3 = (_RR(_I(s)) >> (s.H + 1));
		auto d2 = (_R(_I(s)) >> (s.H + 1));

		unsigned int n1 = _N(_I(s));
		unsigned int n2 = (n1 + 1) % 3;
		unsigned int n3 = (n2 + 1) % 3;

		return (

		(f[s + d2] * g_(s + d2, 1L << n2) - f[s - d2] * g_(s - d2, 1L << n2)) * inv_L[n2] * idh(s, n2) -

		(f[s + d3] * g_(s + d3, 1L << n3) - f[s - d3] * g_(s - d3, 1L << n3)) * inv_L[n3] * idh(s, n3)

		) / g_(s, ~(1L << n1));
	}

	template<int N, typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,
	        Field<Geometry<this_type, N>, TL> const & f, index_type s);

	template<int IL, int IR, typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,
	        Field<Geometry<this_type, IL>, TL> const &l, Field<Geometry<this_type, IR>, TR> const &r,
	        index_type s) const;

	template<int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>, Field<Geometry<this_type, IL>, TL> const & f,
	        index_type s) const;

	template<typename TL, typename TR> inline auto OpEval(Int2Type<DOT>, Field<Geometry<this_type, EDGE>, TL> const &l,
	        Field<Geometry<this_type, EDGE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto d1 = _MI >> (s.H + 1);
		auto d2 = _MJ >> (s.H + 1);
		auto d3 = _MK >> (s.H + 1);
		return (

		(l[s + d1] + l[s - d1]) * (l[s + d1] + l[s - d1]) +

		(l[s + d2] + l[s - d2]) * (r[s + d2] + r[s - d2]) +

		(l[s + d3] + l[s - d3]) * (r[s + d3] + r[s - d3])) * 0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<DOT>, Field<Geometry<this_type, FACE>, TL> const &l,
	        Field<Geometry<this_type, FACE>, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto d1 = (_MJ >> (s.H + 1)) | (_MK >> (s.H + 1));
		auto d2 = (_MK >> (s.H + 1)) | (_MI >> (s.H + 1));
		auto d3 = (_MI >> (s.H + 1)) | (_MJ >> (s.H + 1));
		return (

		(l[s + d1] + l[s - d1]) * (l[s + d1] + l[s - d1]) +

		(l[s + d2] + l[s - d2]) * (r[s + d2] + r[s - d2]) +

		(l[s + d3] + l[s - d3]) * (r[s + d3] + r[s - d3])) * 0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<CROSS>,
	        Field<Geometry<this_type, EDGE>, TL> const &l, Field<Geometry<this_type, FACE>, TR> const &r,
	        index_type s) const ->decltype(l[s]*r[s])
	{

	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<CROSS>,
	        Field<Geometry<this_type, VERTEX>, TL> const &l, Field<Geometry<this_type, VERTEX>, TR> const &r,
	        index_type s) const
	        DECL_RET_TYPE( ( Cross(l[s],r[s]) ))
};

}
// namespace simpla

#endif /* RECT_MESH_H_ */

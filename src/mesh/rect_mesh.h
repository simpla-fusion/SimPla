/*
 * rect_mesh.h
 *
 *  Created on: 2014年2月26日
 *      Author: salmon
 */

#ifndef RECT_MESH_H_
#define RECT_MESH_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <memory>

#include "../fetl/field.h"
#include "../fetl/ntuple.h"
#include "../fetl/primitives.h"
#include "../modeling/media_tag.h"
#include "../physics/physical_constants.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/type_utilites.h"
//#include "../utilities/utilities.h"
#include "../utilities/memory_pool.h"
#include "octree_forest.h"

namespace simpla
{

struct EuclideanSpace
{
	static constexpr int NDIMS = 3;
	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;
	typedef nTuple<NDIMS, Real> coordinates_type;

	static constexpr Real g_t[NDIMS][NDIMS] = {

	1, 0, 0,

	0, 1, 0,

	0, 0, 1

	};

	//! diagonal term of metric tensor
	template<typename TI>
	constexpr Real v(TI const & s) const
	{
		return 1.0;
	}
	//! diagonal term of metric tensor
	template<typename TI>
	constexpr Real l_v(TI const & s) const
	{
		return 1.0;
	}

	template<typename index_type>
	vector_type PullBack(index_type const & s, vector_type const & v) const
	{
		return v;
	}

	template<typename index_type>
	vector_type PushForward(index_type const & s, vector_type const & v) const
	{
		return v;
	}

	template<typename index_type>
	vector_type PullBack(coordinates_type const & x, vector_type const & v) const
	{
		return v;
	}

	template<typename index_type>
	vector_type PushForward(coordinates_type const & x, vector_type const & v) const
	{
		return v;
	}

};
template<typename Metric = EuclideanSpace>
class RectMesh: public OcForest, public Metric
{
public:
	typedef OcForest base_type;
	typedef RectMesh this_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr int NUM_OF_COMPONENT_TYPE = NDIMS + 1;
	typedef typename OcForest::index_type index_type;

	RectMesh()
			: tags_(*this)
	{
		;
	}
	~RectMesh()
	{
		;
	}

	template<typename TDict>
	RectMesh(TDict const & dict)
			: OcForest(dict), tags_(*this)
	{
		Load(dict);
	}

	this_type & operator=(const this_type&) = delete;

	void swap(this_type & rhs)
	{
		OcForest::swap(rhs);
	}

	template<typename TDict>
	void Load(TDict const & dict)
	{
	}

	std::ostream & Save(std::ostream &os) const
	{
		return os;
	}

	void Update()
	{

	}

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************

	typedef Real scalar_type;

	//* Container: storage depend

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (GetNumOfElements(iform)));
	}

	//* Media Tags

	MediaTag<this_type> tags_;

	typedef typename MediaTag<this_type>::tag_type tag_type;
	MediaTag<this_type> & tags()
	{
		return tags_;
	}
	MediaTag<this_type> const& tags() const
	{

		return tags_;
	}

	nTuple<NDIMS,Real> dx_;
	nTuple<NDIMS,Real> const & GetDx()const
	{
		return dx_;
	}

	//	//* Time
//
//	Real dt_ = 0.0;//!< time step
//	Real time_ = 0.0;
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
//		res += inv_dx_[i] * inv_dx_[i];
//
//		return std::sqrt(res) * speed_of_light * dt_;
//	}
//
//	void FixCourant(Real a)
//	{
//		dt_ *= a / CheckCourant();
//	}

//***************************************************************************************************
// Geometric properties
// Metric
//***************************************************************************************************

	typedef nTuple<3, Real> coordinates_type;

	coordinates_type xmin_ =
	{	0, 0, 0};

	coordinates_type xmax_ =
	{	1, 1, 1};

	coordinates_type inv_L =
	{	1.0, 1.0, 1.0};

	template<int IN, typename T>
	inline void SetExtent(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NDIMS ? IN : NDIMS;

		for (int i = 0; i < n; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];
		}

		for (int i = n; i < NDIMS; ++i)
		{
			xmin_[i] = 0;
			xmax_[i] = 0;
		}
	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline coordinates_type GetCoordinates(index_type const &s) const
	{
		coordinates_type res;
		res = xmin_ + (xmax_ - xmin_) * base_type::GetCoordinates(s);
		return std::move(res);
	}

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return l[s] * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, EDGE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = _D(s);
		return ((l[s - X] + l[s + X]) * 0.5 * r[s]);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, FACE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y = _D(_R(_I(s)) );
		auto Z = _D(_RR(_I(s)) );

		return (l[(s - Y) - Z] + l[(s - Y) + Z] + l[(s + Y) - Z] + l[(s + Y) + Z]) * 0.25 * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, VOLUME, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = _DI >> (H(s) + 1);
		auto Y = _DJ >> (H(s) + 1);
		auto Z = _DK >> (H(s) + 1);

		return (

		l[((s - X) - Y) - Z] + l[((s - X) - Y) + Z] + l[((s - X) + Y) - Z] + l[((s - X) + Y) + Z] +

		l[((s + X) - Y) - Z] + l[((s + X) - Y) + Z] + l[((s + X) + Y) - Z] + l[((s + X) + Y) + Z]

		) * 0.125 * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = _D(s );
		return l[s]*(r[s-X]+r[s+X])*0.5;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, EDGE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y = _D(_R(_I(s)) );
		auto Z = _D(_RR(_I(s)));

		return ((l[s - Y] + l[s + Y]) * (l[s - Z] + l[s + Z]) * 0.25);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, FACE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));

		return

		(l[(s - Y) - Z] + l[(s - Y) + Z] + l[(s + Y) - Z] + l[(s + Y) + Z]) * (r[s - X] + r[s + X]) * 0.125 +

		(l[(s - Z) - X] + l[(s - Z) + X] + l[(s + Z) - X] + l[(s + Z) + X]) * (r[s - Y] + r[s + Y]) * 0.125 +

		(l[(s - X) - Y] + l[(s - X) + Y] + l[(s + X) - Y] + l[(s + X) + Y]) * (r[s - Z] + r[s + Z]) * 0.125;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, FACE, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y =_D( _R(_I(s)) );
		auto Z =_D( _RR(_I(s)) );

		return
		l[s]*(

		r[(s-Y)-Z]+
		r[(s-Y)+Z]+
		r[(s+Y)-Z]+
		r[(s+Y)+Z]

		)*0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, FACE, TL> const &l,
	Field<this_type, EDGE, TR> const &r, index_type s) const ->decltype(r[s]*l[s])
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));

		return

		(r[(s - Y) - Z] + r[(s - Y) + Z] + r[(s + Y) - Z] + r[(s + Y) + Z]) * (l[s - X] + l[s + X]) * 0.125 +

		(r[(s - Z) - X] + r[(s - Z) + X] + r[(s + Z) - X] + r[(s + Z) + X]) * (l[s - Y] + l[s + Y]) * 0.125 +

		(r[(s - X) - Y] + r[(s - X) + Y] + r[(s + X) - Y] + r[(s + X) + Y]) * (l[s - Z] + l[s + Z]) * 0.125;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VOLUME, TL> const &l,
	Field<this_type,VERTEX , TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = _DI >> (H(s) + 1);
		auto Y = _DJ >> (H(s) + 1);
		auto Z = _DK >> (H(s) + 1);

		return (

		l[((s - X) - Y) - Z] + l[((s - X) - Y) + Z] + l[((s - X) + Y) - Z] + l[((s - X) + Y) + Z] +

		l[((s + X) - Y) - Z] + l[((s + X) - Y) + Z] + l[((s + X) + Y) - Z] + l[((s + X) + Y) + Z]

		) * 0.125 * r[s];
	}

//***************************************************************************************************

	template<int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>,Field<this_type, IL , TL> const & f,
	index_type s) const-> decltype(f[s]+f[s])
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));
		return

		(

		f[((s + X) - Y) - Z] + f[((s + X) - Y) + Z] + f[((s + X) + Y) - Z] + f[((s + X) + Y) + Z] +

		f[((s - X) - Y) - Z] + f[((s - X) - Y) + Z] + f[((s - X) + Y) - Z] + f[((s - X) + Y) + Z]

		) * 0.125;
	}

//***************************************************************************************************

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, VERTEX, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto d = _D( s );
		return (f[s + d] - f[s - d]);
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, EDGE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = _D(_I(s));
		auto Y = _R(X);
		auto Z = _RR(X);

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, FACE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));

		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<int IL, typename TL> void OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, IL , TL> const & f,
	index_type s)const = delete;

	template<int IL, typename TL> void OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, IL , TL> const & f,
	index_type s) const= delete;

	template< typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, EDGE, TL> const & f,
	index_type s)const->typename std::remove_reference<decltype(f[s])>::type
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));

		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, FACE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = _D(s);
		auto Y = _R(X);
		auto Z = _RR(X);
		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}
	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, VOLUME, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto d = _D( _I(s) );
		return (f[s + d] - f[s - d]);
	}

	template<typename TL, typename TR> void OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, VERTEX, TL> const & f, index_type s) const=delete;

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, EDGE, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		auto X = (_DI >> (H(s) + 1));
		auto Y = (_DJ >> (H(s) + 1));
		auto Z = (_DK >> (H(s) + 1));

		return

		(f[s + X] - f[s - X]) * 0.5 * v[0] +

		(f[s + Y] - f[s - Y]) * 0.5 * v[1] +

		(f[s + Z] - f[s - Z]) * 0.5 * v[2];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, FACE, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		unsigned int n = _C(s);

		auto X = _D(s);
		auto Y = _R(X);
		auto Z = _RR(Y);
		return

		(f[s + Y] + f[s - Y]) * 0.5 * v[(n + 2) % 3] -

		(f[s + Z] + f[s - Z]) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, VOLUME, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		unsigned int n = _C(_I(s));
		unsigned int D = _D( _I(s));

		return (f[s + D] - f[s - D]) * 0.5 * v[n];
	}

};
template<typename TS> inline std::ostream &
operator<<(std::ostream & os, RectMesh<TS> const & d)
{
	d.Save(os);
	return os;
}
}
// namespace simpla

#endif /* RECT_MESH_H_ */

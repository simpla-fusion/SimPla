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
#include "../physics/physical_constants.h"
#include "../utilities/type_utilites.h"
//#include "../utilities/utilities.h"
#include "../utilities/memory_pool.h"

namespace simpla
{
template<typename Topology>
struct EuclideanSpace
{
	typedef Topology topology_type;

	typedef EuclideanSpace<topology_type> this_type;

	static constexpr int NDIMS = topology_type::NDIMS;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;

	typedef nTuple<NDIMS, Real> vector_type;
	typedef nTuple<NDIMS, Real> covector_type;

	topology_type const & topology;

	EuclideanSpace(this_type const & rhs) = delete;

	EuclideanSpace(topology_type const & t)
			: topology(t)
	{

	}
	template<typename TDict>
	EuclideanSpace(topology_type const & t, TDict const & dict)
			: topology(t)
	{

	}

	~EuclideanSpace()
	{

	}

	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	void Update()
	{

	}
	template<typename TDict>
	void Load(TDict const & dict)
	{

	}

	std::ostream & Save(std::ostream & os) const
	{
		return os;
	}

	coordinates_type xmin_ = { 0, 0, 0 };

	coordinates_type xmax_ = { 1, 1, 1 };

	coordinates_type inv_L = { 1.0, 1.0, 1.0 };

	coordinates_type scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type inv_scale_ = { 1.0, 1.0, 1.0 };

	coordinates_type shift_ = { 0, 0, 0 };

	static constexpr nTuple<NDIMS, Real> normal_[NDIMS] = {

	1, 0, 0,

	0, 1, 0,

	0, 0, 1

	};

	Real volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	Real inv_volume_[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	nTuple<NDIMS, Real> dx_;

	nTuple<NDIMS, Real> const & GetDx() const
	{
		return dx_;
	}

	template<int IN, typename T>
	inline void SetExtent(nTuple<IN, T> const & pmin, nTuple<IN, T> const & pmax)
	{
		int n = IN < NDIMS ? IN : NDIMS;

		for (int i = 0; i < n; ++i)
		{
			xmin_[i] = pmin[i];
			xmax_[i] = pmax[i];

			shift_[i] = pmin[i];

			scale_[i] = (pmax[i] > pmin[i]) ? (1.0 / (pmax[i] - pmin[i])) : 0;
			inv_scale_[i] = (pmax[i] - pmin[i]);
		}

	}

	inline std::pair<coordinates_type, coordinates_type> GetExtent() const
	{
		return std::move(std::make_pair(xmin_, xmax_));
	}

	inline coordinates_type GetCoordinates(coordinates_type const &x) const
	{
		return coordinates_type( {

		xmin_[0] + (xmax_[0] - xmin_[0]) * x[0],

		xmin_[1] + (xmax_[1] - xmin_[1]) * x[1],

		xmin_[2] + (xmax_[2] - xmin_[2]) * x[2]

		});
	}

	nTuple<3, Real> const& Normal(index_type s) const
	{
		return normal_[topology.topology_type::_C(s)];
	}

	template<typename TV>
	TV const& Normal(index_type s, nTuple<3, TV> const & v) const
	{
		return v[topology.topology_type::_C(s)];
	}

	Real const& Volume(index_type s) const
	{
		return volume_[topology._N(s)];
	}
	Real const& InvVolume(index_type s) const
	{
		return inv_volume_[topology._N(s)];
	}
	coordinates_type Trans(coordinates_type const &x) const
	{
		return coordinates_type( {

		(x[0] - shift_[0]) * scale_[0],

		(x[0] - shift_[1]) * scale_[1],

		(x[0] - shift_[2]) * scale_[2]

		});
	}
	coordinates_type InvTrans(coordinates_type const &x) const
	{
		return coordinates_type( {

		x[0] * inv_scale_[0] + shift_[0],

		x[1] * inv_scale_[1] + shift_[1],

		x[2] * inv_scale_[2] + shift_[2]

		});
	}
}
;

class OcForest;
/**
 *  Grid is mapped as a rectangle region;
 *
 */
template<typename TTopology = OcForest, template<typename > class Geometry = EuclideanSpace>
class RectMesh: public TTopology, public Geometry<TTopology>
{
public:
	typedef RectMesh<TTopology, Geometry> this_type;
	typedef TTopology topology_type;
	typedef Geometry<topology_type> geometry_type;

	typedef Real scalar_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;

	template<typename ... Args>
	RectMesh(Args const &... args)
			: geometry_type(static_cast<TTopology const &>(*this)), dt_(1.0), time_(0.0)
	{
		Load(std::forward<Args const &>(args)...);
	}
	~RectMesh()
	{
	}
	RectMesh(const this_type&) = delete;
	this_type & operator=(const this_type&) = delete;

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	void Load()
	{
	}

	template<typename ... Args>
	void Load(Args const &... args)
	{
		topology_type::Load(std::forward<Args const &>(args)...);
		geometry_type::Load(std::forward<Args const &>(args)...);
	}

	std::ostream & Save(std::ostream &os) const
	{
		topology_type::Save(os);
		geometry_type::Save(os);
		return os;
	}

	void Update()
	{
		topology_type::Update();
		geometry_type::Update();
	}

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************
	//* Container: storage depend

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (topology_type::GetNumOfElements(iform)));
	}

	PhysicalConstants constants_;

	PhysicalConstants & constants()
	{
		return constants_;
	}

	PhysicalConstants const & constants()const
	{
		return constants_;
	}

	//* Time

	Real dt_ = 0.0;//!< time step
	Real time_ = 0.0;

	void NextTimeStep()
	{
		time_ += dt_;
	}
	Real GetTime() const
	{
		return time_;
	}

	void GetTime(Real t)
	{
		time_ = t;
	}
	inline Real GetDt() const
	{
		CHECK(CheckCourant());
		return dt_;
	}

	inline void SetDt(Real dt = 0.0)
	{
		dt_ = dt;
		Update();
	}
	double CheckCourant() const
	{
		DEFINE_GLOBAL_PHYSICAL_CONST

		nTuple<3, Real> inv_dx_;
		Real res = 0.0;

		for (int i = 0; i < 3; ++i)
		res += inv_dx_[i] * inv_dx_[i];

		return std::sqrt(res) * speed_of_light * dt_;
	}

	void FixCourant(Real a=1.0)
	{
		dt_ *= a / CheckCourant();
	}

	inline auto GetExtent() const
	DECL_RET_TYPE(geometry_type::GetExtent())

	//***************************************************************************************************
	// Cell-wise operation
	//***************************************************************************************************

	template<int IFORM, typename TV>
	TV Sample(Int2Type<IFORM>,index_type s, TV const & v) const
	{
		return v * geometry_type::Volume(s);
	}

	template<int IFORM, typename TV>
	typename std::enable_if<(IFORM== EDGE || IFORM == FACE),TV>::type
	Sample(Int2Type<IFORM>,index_type s, nTuple<NDIMS, TV> const & v) const
	{
		return geometry_type::Normal( s , v) * geometry_type::Volume(s);
	}

//	template<int IFORM,typename TV>
//	typename Field<this_type,IFORM,TV>::field_value_type Get(Field<this_type,IFORM,TV> const & f,coordiantes_type const &x )
//	{
//		return topology_type::Get(f,geometry_type::Trans(x));
//	}

	coordinates_type GetCoordinates(index_type s)const
	{
		return std::move(geometry_type::InvTrans(topology_type::GetCoordinates(s)));
	}

	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, VERTEX, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto d = topology_type::_D( s );
		return (f[s + d] - f[s - d]);
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, EDGE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = topology_type::_D(topology_type::_I(s));
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, FACE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<int IL, typename TL> void OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, IL , TL> const & f,
	index_type s)const = delete;

	template<int IL, typename TL> void OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, IL , TL> const & f,
	index_type s) const= delete;

	template< typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, EDGE, TL> const & f,
	index_type s)const->decltype(f[s]-f[s])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));
		return (f[s + X] - f[s - X]) + (f[s + Y] - f[s - Y]) + (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, FACE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = topology_type::_D(s);
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, VOLUME, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto d = topology_type::_D( topology_type::_I(s) );

		return (f[s + d] - f[s - d]);
	}
	//***************************************************************************************************

	//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		return l[s] * r[s] * geometry_type::InvVolume(s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, EDGE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = topology_type::_D(s);
		return
		(

		l[s - X]*geometry_type::InvVolume(s-X) +

		l[s + X]*geometry_type::InvVolume(s+X)

		) * 0.5 * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, FACE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = topology_type::_D(topology_type::_I(s));
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);

		return (

		l[(s - Y) - Z]*geometry_type::InvVolume((s - Y) - Z) +

		l[(s - Y) + Z]*geometry_type::InvVolume((s - Y) + Z) +

		l[(s + Y) - Z]*geometry_type::InvVolume((s + Y) - Z) +

		l[(s + Y) + Z]*geometry_type::InvVolume((s + Y) + Z)

		) * 0.25 * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VERTEX, TL> const &l,
	Field<this_type, VOLUME, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = topology_type::_DI >> (topology_type::H(s) + 1);
		auto Y = topology_type::_DJ >> (topology_type::H(s) + 1);
		auto Z = topology_type::_DK >> (topology_type::H(s) + 1);

		return (

		l[((s - X) - Y) - Z]*geometry_type::InvVolume(((s - X) - Y) - Z) +

		l[((s - X) - Y) + Z]*geometry_type::InvVolume(((s - X) - Y) + Z) +

		l[((s - X) + Y) - Z]*geometry_type::InvVolume(((s - X) + Y) - Z) +

		l[((s - X) + Y) + Z]*geometry_type::InvVolume(((s - X) + Y) + Z) +

		l[((s + X) - Y) - Z]*geometry_type::InvVolume(((s + X) - Y) - Z) +

		l[((s + X) - Y) + Z]*geometry_type::InvVolume(((s + X) - Y) + Z) +

		l[((s + X) + Y) - Z]*geometry_type::InvVolume(((s + X) + Y) - Z) +

		l[((s + X) + Y) + Z]*geometry_type::InvVolume(((s + X) + Y) + Z)

		) * 0.125 * r[s];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = topology_type::_D(s );
		return l[s]*(r[s-X]+r[s+X])*0.5;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, EDGE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y = topology_type::_D(topology_type::_R(topology_type::_I(s)) );
		auto Z = topology_type::_D(topology_type::_RR(topology_type::_I(s)));

		return (

		(
				l[s - Y]*geometry_type::InvVolume(s-Y)+

				l[s + Y]*geometry_type::InvVolume(s+Y)

		) * (
				l[s - Z]*geometry_type::InvVolume(s-Z)+

				l[s + Z]*geometry_type::InvVolume(s+Z)

		) * 0.25*geometry_type:: Volume(s ));
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, EDGE, TL> const &l,
	Field<this_type, FACE, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return

		((

				l[(s - Y) - Z]*geometry_type::InvVolume((s - Y) - Z) +

				l[(s - Y) + Z]*geometry_type::InvVolume((s - Y) + Z) +

				l[(s + Y) - Z]*geometry_type::InvVolume((s + Y) - Z) +

				l[(s + Y) + Z]*geometry_type::InvVolume((s + Y) + Z)

		) * (

				r[s - X]*geometry_type::InvVolume(s - X) +

				r[s + X]*geometry_type::InvVolume(s + X)

		) + (

				l[(s - Z) - X]*geometry_type::InvVolume((s - Z) - X) +

				l[(s - Z) + X]*geometry_type::InvVolume((s - Z) + X) +

				l[(s + Z) - X]*geometry_type::InvVolume((s + Z) - X) +

				l[(s + Z) + X]*geometry_type::InvVolume((s + Z) + X)

		) * (

				r[s - Y]*geometry_type::InvVolume(s - Y) +

				r[s + Y]*geometry_type::InvVolume(s + Y)

		) + (

				l[(s - X) - Y]*geometry_type::InvVolume((s - X) - Y) +

				l[(s - X) + Y]*geometry_type::InvVolume((s - X) + Y) +

				l[(s + X) - Y]*geometry_type::InvVolume((s + X) - Y) +

				l[(s + X) + Y]*geometry_type::InvVolume((s + X) + Y)

		) * (

				r[s - Z]*geometry_type::InvVolume(s - Z) +

				r[s + Z]*geometry_type::InvVolume(s + Z)

		) )* 0.125*geometry_type::Volume(s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, FACE, TL> const &l,
	Field<this_type, VERTEX, TR> const &r, index_type s) const ->decltype(l[s]*r[s])
	{
		auto Y =topology_type::_D( topology_type::_R(topology_type::_I(s)) );
		auto Z =topology_type::_D( topology_type::_RR(topology_type::_I(s)) );

		return
		l[s]*(

		r[(s-Y)-Z]*geometry_type::InvVolume((s - Y) - Z)+
		r[(s-Y)+Z]*geometry_type::InvVolume((s - Y) + Z)+
		r[(s+Y)-Z]*geometry_type::InvVolume((s + Y) - Z)+
		r[(s+Y)+Z]*geometry_type::InvVolume((s + Y) + Z)

		)*0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, FACE, TL> const &r,
	Field<this_type, EDGE, TR> const &l, index_type s) const ->decltype(l[s]*r[s])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return

		((

				r[(s - Y) - Z]*geometry_type::InvVolume((s - Y) - Z) +

				r[(s - Y) + Z]*geometry_type::InvVolume((s - Y) + Z) +

				r[(s + Y) - Z]*geometry_type::InvVolume((s + Y) - Z) +

				r[(s + Y) + Z]*geometry_type::InvVolume((s + Y) + Z)

		) * (

				l[s - X]*geometry_type::InvVolume(s - X) +

				l[s + X]*geometry_type::InvVolume(s + X)

		) + (

				r[(s - Z) - X]*geometry_type::InvVolume((s - Z) - X) +

				r[(s - Z) + X]*geometry_type::InvVolume((s - Z) + X) +

				r[(s + Z) - X]*geometry_type::InvVolume((s + Z) - X) +

				r[(s + Z) + X]*geometry_type::InvVolume((s + Z) + X)

		) * (

				l[s - Y]*geometry_type::InvVolume(s - Y) +

				l[s + Y]*geometry_type::InvVolume(s + Y)

		) + (

				r[(s - X) - Y]*geometry_type::InvVolume((s - X) - Y) +

				r[(s - X) + Y]*geometry_type::InvVolume((s - X) + Y) +

				r[(s + X) - Y]*geometry_type::InvVolume((s + X) - Y) +

				r[(s + X) + Y]*geometry_type::InvVolume((s + X) + Y)

		) * (

				l[s - Z]*geometry_type::InvVolume(s - Z) +

				l[s + Z]*geometry_type::InvVolume(s + Z)

		) )* 0.125*geometry_type::Volume(s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>,Field<this_type, VOLUME, TL> const &l,
	Field<this_type,VERTEX , TR> const &r, index_type s) const ->decltype(r[s]*l[s])
	{
		auto X = topology_type::_DI >> (topology_type::H(s) + 1);
		auto Y = topology_type::_DJ >> (topology_type::H(s) + 1);
		auto Z = topology_type::_DK >> (topology_type::H(s) + 1);

		return

		l[s] *(

		r[((s - X) - Y) - Z]*geometry_type::InvVolume(((s - X) - Y) - Z) +

		r[((s - X) - Y) + Z]*geometry_type::InvVolume(((s - X) - Y) + Z) +

		r[((s - X) + Y) - Z]*geometry_type::InvVolume(((s - X) + Y) - Z) +

		r[((s - X) + Y) + Z]*geometry_type::InvVolume(((s - X) + Y) + Z) +

		r[((s + X) - Y) - Z]*geometry_type::InvVolume(((s + X) - Y) - Z)+

		r[((s + X) - Y) + Z]*geometry_type::InvVolume(((s + X) - Y) + Z) +

		r[((s + X) + Y) - Z]*geometry_type::InvVolume(((s + X) + Y) - Z) +

		r[((s + X) + Y) + Z]*geometry_type::InvVolume(((s + X) + Y) + Z)

		) * 0.125;
	}

//***************************************************************************************************

	template<int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>,Field<this_type, IL , TL> const & f,
	index_type s) const-> decltype(f[s]+f[s])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return

		(

		f[((s + X) - Y) - Z]*geometry_type::InvVolume(((s + X) - Y) - Z) +

		f[((s + X) - Y) + Z]*geometry_type::InvVolume(((s + X) - Y) + Z) +

		f[((s + X) + Y) - Z]*geometry_type::InvVolume(((s + X) + Y) - Z) +

		f[((s + X) + Y) + Z]*geometry_type::InvVolume(((s + X) + Y) + Z) +

		f[((s - X) - Y) - Z]*geometry_type::InvVolume(((s - X) - Y) - Z)+

		f[((s - X) - Y) + Z]*geometry_type::InvVolume(((s - X) - Y) + Z) +

		f[((s - X) + Y) - Z]*geometry_type::InvVolume(((s - X) + Y) - Z) +

		f[((s - X) + Y) + Z]*geometry_type::InvVolume(((s - X) + Y) + Z)

		) * 0.125 * geometry_type::Volume(s);
	}

	template<typename TL, typename TR> void OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, VERTEX, TL> const & f, index_type s) const=delete;

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, EDGE, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return

		(f[s + X] - f[s - X]) * 0.5 * v[0] +

		(f[s + Y] - f[s - Y]) * 0.5 * v[1] +

		(f[s + Z] - f[s - Z]) * 0.5 * v[2];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, FACE, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		unsigned int n = topology_type::_C(s);

		auto X = topology_type::_D(s);
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(Y);
		return

		(f[s + Y] + f[s - Y]) * 0.5 * v[(n + 2) % 3] -

		(f[s + Z] + f[s - Z]) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>,nTuple<NDIMS, TR> const & v,
	Field<this_type, VOLUME, TL> const & f, index_type s)const->decltype(f[s]*v[0])
	{
		unsigned int n = topology_type::_C(topology_type::_I(s));
		unsigned int D = topology_type::_D(topology_type::_I(s));

		return (f[s + D] - f[s - D]) * 0.5 * v[n];
	}

};
template<typename TTopology, template<typename > class TGeo> inline std::ostream &
operator<<(std::ostream & os, RectMesh<TTopology, TGeo> const & d)
{
	return d.Save(os);
}
}
// namespace simpla

#endif /* RECT_MESH_H_ */

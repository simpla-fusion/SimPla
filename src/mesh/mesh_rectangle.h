/*
 * mesh_rectangle.h
 *
 *  Created on: 2014年2月26日
 *      Author: salmon
 */

#ifndef MESH_RECTANGLE_H_
#define MESH_RECTANGLE_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../fetl/fetl.h"
#include "../utilities/memory_pool.h"
#include "../utilities/type_utilites.h"
#include "interpolator.h"
#include "glaobal_mesh.h"
namespace simpla
{
template<typename > class EuclideanGeometry;

class OcForest;
/**
 *  Grid is mapped as a rectangle/hexahedrom Range;
 *
 */
template<typename TTopology = OcForest, template<typename > class Geometry = EuclideanGeometry>
class RectMesh: public TTopology, public Geometry<TTopology>
{
public:
	typedef RectMesh<TTopology, Geometry> this_type;
	typedef TTopology topology_type;
	typedef Geometry<topology_type> geometry_type;

	typedef Interpolator<this_type> interpolator_type;

	typedef Real scalar_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::compact_index_type compact_index_type;

	//* Time

	Real dt_ = 0.0; //!< time step
	Real time0_ = 0.0;
	unsigned long clock_ = 0UL;

	RectMesh()
			: geometry_type(static_cast<TTopology const &>(*this))
	{
	}

	template<typename TDict>
	RectMesh(TDict const & dict)
			: geometry_type(static_cast<TTopology const &>(*this))
	{
		Load(dict);
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

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const &...others)
	{
		LOGGER << "Load Mesh RectMesh";
		topology_type::Load(dict["Topology"], std::forward<Others const&>(others)...);
		geometry_type::Load(dict["Geometry"], std::forward<Others const&>(others)...);

		dt_ = dict["dt"].template as<Real>(1.0);

	}

	std::string Save(std::string const &path) const
	{
		std::stringstream os;

		os

		<< "\n Topology  = { " << topology_type::Save(path) << "}, "

		<< "\n Geometry  = { " << geometry_type::Save(path) << "},"

		<< "\n dt=" << dt_;

		return os.str();
	}

	//***************************************************************************************************

	template<typename TI>
	void Decompose(TI const &num_process, TI const & process_num, unsigned int gw = 2)
	{

		auto extent = topology_type::Decompose(num_process, process_num, gw);

		geometry_type::SetExtent(GetCoordinates(extent.first), GetCoordinates(extent.second));

	}

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************

	template<typename TV> using Container=std::shared_ptr<TV>;

	template<int iform, typename TV> inline std::shared_ptr<TV> MakeContainer() const
	{
		return (MEMPOOL.allocate_shared_ptr < TV > (topology_type::GetNumOfElements(iform)));
	}

	// Time
	void NextTimeStep()
	{
		++clock_;
	}
	Real GetTime() const
	{
		return static_cast<double>(clock_)*dt_+time0_;
	}

	Real GetDt() const
	{
		return dt_;
	}

	Real CheckCourantDt(nTuple<3,Real> const & u) const
	{

		Real dt = dt_;
		auto dims = topology_type::GetDimensions();
		auto extent = geometry_type::GetExtent();

		Real r = 0.0;
		for (int s = 0; s < 3; ++s)
		{
			if (dims[s] > 1)
			{
				r += u[s] / (extent.second[s] - extent.first[s]);
			}
		}

		if (dt * r > 1.0)
		{
			dt = 0.5 / r;
		}

		return dt;
	}

	Real CheckCourantDt(Real speed) const
	{
		return CheckCourantDt(nTuple<3,Real>(
				{	speed,speed,speed}));
	}

//***************************************************************************************************

	Real Volume(index_type s) const
	{
		return geometry_type::Volume(s);
	}
	Real InvVolume(index_type s) const
	{
		return geometry_type::InvVolume(s);
	}

	Real DualVolume(index_type s) const
	{
		return geometry_type::DualVolume(s);
	}
	Real InvDualVolume(index_type s) const
	{
		return geometry_type::InvDualVolume(s);
	}

	coordinates_type CoordinatesLocalToGlobal(index_type s, coordinates_type x) const
	{
		return geometry_type::CoordinatesLocalToGlobal(topology_type::CoordinatesLocalToGlobal(s,x));
	}
	index_type CoordinatesGlobalToLocal(coordinates_type *x,compact_index_type shift=0 )const
	{
		*x=geometry_type::CoordinatesGlobalToLocal(*x);
		return topology_type::CoordinatesGlobalToLocal(x,shift );
	}
	index_type CoordinatesGlobalToLocalDual(coordinates_type *x,compact_index_type shift=0)const
	{
		*x=geometry_type::CoordinatesGlobalToLocal(*x);
		return topology_type::CoordinatesGlobalToLocalDual(x,shift);
	}
	index_type GetCellIndex(coordinates_type x )const
	{
		auto r=geometry_type::CoordinatesGlobalToLocal( x);
		return topology_type::CoordinatesGlobalToLocal(&r);

	}

	index_type GetCellIndex(index_type s)const
	{
		return topology_type::GetCellIndex(s);

	}

	template<typename TI>
	coordinates_type GetCoordinates(TI s) const
	{
		return geometry_type::CoordinatesLocalToGlobal(topology_type::GetCoordinates(s));
	}

	template<int IFORM,typename TExpr>
	inline typename Field<this_type,IFORM,TExpr>::field_value_type
	Gather(Field<this_type,IFORM,TExpr>const &f,coordinates_type x ) const
	{
		return std::move(interpolator_type::Gather(f,x));

	}

	template<int IFORM, typename TExpr>
	inline void Scatter( coordinates_type x ,
	typename Field<this_type,IFORM,TExpr>::field_value_type const & v ,Field<this_type,IFORM,TExpr> *f )const
	{
		interpolator_type::Scatter(x,v,f );
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
		auto X = topology_type::_D(topology_type::_Dual(s));
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
		auto a= geometry_type::InvDualVolume(s)*geometry_type::Volume(s);
		return
		-((
				f[s + X]*(geometry_type::InvVolume(s+X)*geometry_type::DualVolume(s+X)*a)-
				f[s - X]*(geometry_type::InvVolume(s-X)*geometry_type::DualVolume(s-X)*a)

		) + (
				f[s + Y]*(geometry_type::InvVolume(s+Y)*geometry_type::DualVolume(s+Y)*a)-
				f[s - Y]*(geometry_type::InvVolume(s-Y)*geometry_type::DualVolume(s-Y)*a)

		) + (
				f[s + Z]*(geometry_type::InvVolume(s+Z)*geometry_type::DualVolume(s+Z)*a) -
				f[s - Z]*(geometry_type::InvVolume(s-Z)*geometry_type::DualVolume(s-Z)*a)

		) )
		;

	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, FACE, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto X = topology_type::_D(s);
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);
		Real a= geometry_type::InvDualVolume(s) *geometry_type::Volume(s);
		return (

		f[s + Y]*(geometry_type::InvVolume(s + Y)*geometry_type::DualVolume(s + Y)*a)

		-f[s - Y]*(geometry_type::InvVolume(s - Y)*geometry_type::DualVolume(s - Y)*a)

		-f[s + Z]*(geometry_type::InvVolume(s + Z)*geometry_type::DualVolume(s + Z)*a)

		+f[s - Z]*(geometry_type::InvVolume(s - Z)*geometry_type::DualVolume(s - Z)*a)

		)
		;
	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, VOLUME, TL> const & f,
	index_type s)const-> decltype(f[s]-f[s])
	{
		auto d = topology_type::_D( topology_type::_Dual(s) );
		auto a=geometry_type::InvDualVolume(s) *geometry_type::Volume(s);
		return
		-(
		f[s + d]*(geometry_type::InvVolume(s + d) *geometry_type::DualVolume(s + d)*a)

		- f[s - d]*(geometry_type::InvVolume(s - d) *geometry_type::DualVolume(s - d)*a)

		);
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
		auto X = topology_type::_D(topology_type::_Dual(s));
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
		auto Y = topology_type::_D(topology_type::_R(topology_type::_Dual(s)) );
		auto Z = topology_type::_D(topology_type::_RR(topology_type::_Dual(s)));

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
		auto Y =topology_type::_D( topology_type::_R(topology_type::_Dual(s)) );
		auto Z =topology_type::_D( topology_type::_RR(topology_type::_Dual(s)) );

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
	index_type s) const-> decltype(f[s]*1.0)
	{
//		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
//		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
//		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));
//
//		return
//
//		(
//
//		f[((s + X) - Y) - Z]*geometry_type::InvVolume(((s + X) - Y) - Z) +
//
//		f[((s + X) - Y) + Z]*geometry_type::InvVolume(((s + X) - Y) + Z) +
//
//		f[((s + X) + Y) - Z]*geometry_type::InvVolume(((s + X) + Y) - Z) +
//
//		f[((s + X) + Y) + Z]*geometry_type::InvVolume(((s + X) + Y) + Z) +
//
//		f[((s - X) - Y) - Z]*geometry_type::InvVolume(((s - X) - Y) - Z) +
//
//		f[((s - X) - Y) + Z]*geometry_type::InvVolume(((s - X) - Y) + Z) +
//
//		f[((s - X) + Y) - Z]*geometry_type::InvVolume(((s - X) + Y) - Z) +
//
//		f[((s - X) + Y) + Z]*geometry_type::InvVolume(((s - X) + Y) + Z)
//
//		) * 0.125 * geometry_type::Volume(s);

		return f[s]*geometry_type::InvVolume(s)*geometry_type::DualVolume(s);
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
		unsigned int n = topology_type::_C(topology_type::_Dual(s));
		unsigned int D = topology_type::_D(topology_type::_Dual(s));

		return (f[s + D] - f[s - D]) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<int N ,typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,Field<this_type, EDGE, TL> const & f,Int2Type<N>,
	index_type s)const-> decltype(f[s]-f[s])
	{

		auto X = topology_type::_D(topology_type::_Dual(s.d));
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);

		Y = (topology_type::_C(Y)==N)?Y:0UL;
		Z = (topology_type::_C(Z)==N)?Z:0UL;

		return (f[s + Y] - f[s - Y]) - (f[s + Z] - f[s - Z]);
	}

	template<int N,typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,Field<this_type, FACE, TL> const & f,Int2Type<N>,
	index_type s)const-> decltype(f[s]-f[s])
	{

		auto X = topology_type::_D(s.d);
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);

		Y = (topology_type::_C(Y)==N)?Y:0UL;
		Z = (topology_type::_C(Z)==N)?Z:0UL;

		Real a= geometry_type::InvDualVolume(s) *geometry_type::Volume(s);
		return (

		f[s + Y]*(geometry_type::InvVolume(s + Y)*geometry_type::DualVolume(s + Y)*a)

		-f[s - Y]*(geometry_type::InvVolume(s - Y)*geometry_type::DualVolume(s - Y)*a)

		-f[s + Z]*(geometry_type::InvVolume(s + Z)*geometry_type::DualVolume(s + Z)*a)

		+f[s - Z]*(geometry_type::InvVolume(s - Z)*geometry_type::DualVolume(s - Z)*a)

		)
		;
	}
	template<int IL, typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<IL> const &,
	Field<this_type, IL, TR> const & f, index_type s)const
	DECL_RET_TYPE(f[s])

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<VERTEX> const &,
	Field<this_type, EDGE, TR> const & f, index_type s)const->nTuple<3,decltype(f[s]*1.0)>
	{

		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return nTuple<3,decltype(f[s]*1.0)>(
		{
			(f[s - X]*geometry_type::InvVolume(s-X) + f[s + X]*geometry_type::InvVolume(s+X))*0.5*geometry_type::Volume(s),

			(f[s - Y]*geometry_type::InvVolume(s-Y) + f[s + Y]*geometry_type::InvVolume(s+Y))*0.5*geometry_type::Volume(s),

			(f[s - Z]*geometry_type::InvVolume(s-Z) + f[s + Z]*geometry_type::InvVolume(s+Z))*0.5*geometry_type::Volume(s)
		}
		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<EDGE>const &,
	Field<this_type, VERTEX, TR> const & f, index_type s)const->decltype(f[s][0]*1.0)
	{

		auto n = topology_type::_C(s);
		auto D = topology_type::_D(s);

		return (
		(f[s - D][n]*geometry_type::InvVolume(s-D) + f[s + D][n]*geometry_type::InvVolume(s+D))*0.5*geometry_type::Volume(s)

		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<VERTEX>const &,
	Field<this_type, FACE, TR> const & f, index_type s)const->nTuple<3,decltype(f[s]*1.0)>
	{

		auto X = ((topology_type::_DI) >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return nTuple<3,decltype(f[s]*1.0)>(
		{
			(
					f[(s-Y) - Z]*geometry_type::InvVolume((s-Y) - Z) +

					f[(s-Y) + Z]*geometry_type::InvVolume((s-Y) + Z)+

					f[(s+Y) - Z]*geometry_type::InvVolume((s+Y) - Z)+

					f[(s+Y) + Z]*geometry_type::InvVolume((s+Y) + Z)

			)*0.25*geometry_type::Volume(s),

			(
					f[(s-Z) - X]*geometry_type::InvVolume((s-Z) - X) +

					f[(s-Z) + X]*geometry_type::InvVolume((s-Z) + X)+

					f[(s+Z) - X]*geometry_type::InvVolume((s+Z) - X)+

					f[(s+Z) + X]*geometry_type::InvVolume((s+Z) + X)

			)*0.25*geometry_type::Volume(s),

			(
					f[(s-X) - Y]*geometry_type::InvVolume((s-X) - Y) +

					f[(s-X) + Y]*geometry_type::InvVolume((s-X) + Y)+

					f[(s+X) - Y]*geometry_type::InvVolume((s+X) - Y)+

					f[(s+X) + Y]*geometry_type::InvVolume((s+X) + Y)

			)*0.25*geometry_type::Volume(s),

		}
		);
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<FACE>const &,
	Field<this_type, VERTEX, TR> const & f, index_type s)const->decltype(f[s][0]*1.0)
	{

		auto n = topology_type::_C(topology_type::_Dual(s));
		auto X = topology_type::_D(topology_type::_Dual(s));
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);
		return (

		(
				f[(s-Y) - Z][n]*geometry_type::InvVolume((s-Y) - Z) +

				f[(s-Y) + Z][n]*geometry_type::InvVolume((s-Y) + Z)+

				f[(s+Y) - Z][n]*geometry_type::InvVolume((s+Y) - Z)+

				f[(s+Y) + Z][n]*geometry_type::InvVolume((s+Y) + Z)

		)*0.25*geometry_type::Volume(s)

		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type< VOLUME>,
	Field<this_type, FACE, TR> const & f, index_type s)const->nTuple<3,decltype(f[s]*1.0)>
	{

		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return nTuple<3,decltype(f[s]*1.0)>(
		{
			(f[s - X]*geometry_type::InvVolume(s-X) + f[s + X]*geometry_type::InvVolume(s+X))*0.5*geometry_type::Volume(s),

			(f[s - Y]*geometry_type::InvVolume(s-Y) + f[s + Y]*geometry_type::InvVolume(s+Y))*0.5*geometry_type::Volume(s),

			(f[s - Z]*geometry_type::InvVolume(s-Z) + f[s + Z]*geometry_type::InvVolume(s+Z))*0.5*geometry_type::Volume(s)
		}
		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<FACE>,
	Field<this_type, VOLUME, TR> const & f, index_type s)const->decltype(f[s][0]*1.0)
	{

		auto n = topology_type::_C(topology_type::_Dual(s));
		auto D = topology_type::_D(topology_type::_Dual(s));

		return (
		(f[s - D][n]*geometry_type::InvVolume(s-D) + f[s + D][n]*geometry_type::InvVolume(s+D))*0.5*geometry_type::Volume(s)

		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<VOLUME>,
	Field<this_type, EDGE, TR> const & f, index_type s)const->nTuple<3,decltype(f[s]*1.0)>
	{

		auto X = (topology_type::_DI >> (topology_type::H(s) + 1));
		auto Y = (topology_type::_DJ >> (topology_type::H(s) + 1));
		auto Z = (topology_type::_DK >> (topology_type::H(s) + 1));

		return nTuple<3,decltype(f[s]*1.0)>(
		{
			(
					f[(s-Y) - Z]*geometry_type::InvVolume((s-Y) - Z) +

					f[(s-Y) + Z]*geometry_type::InvVolume((s-Y) + Z)+

					f[(s+Y) - Z]*geometry_type::InvVolume((s+Y) - Z)+

					f[(s+Y) + Z]*geometry_type::InvVolume((s+Y) + Z)

			)*0.25*geometry_type::Volume(s),

			(
					f[(s-Z) - X]*geometry_type::InvVolume((s-Z) - X) +

					f[(s-Z) + X]*geometry_type::InvVolume((s-Z) + X)+

					f[(s+Z) - X]*geometry_type::InvVolume((s+Z) - X)+

					f[(s+Z) + X]*geometry_type::InvVolume((s+Z) + X)

			)*0.25*geometry_type::Volume(s),

			(
					f[(s-X) - Y]*geometry_type::InvVolume((s-X) - Y) +

					f[(s-X) + Y]*geometry_type::InvVolume((s-X) + Y)+

					f[(s+X) - Y]*geometry_type::InvVolume((s+X) - Y)+

					f[(s+X) + Y]*geometry_type::InvVolume((s+X) + Y)

			)*0.25*geometry_type::Volume(s),

		}
		);
	}

	template< typename TR> inline auto OpEval(Int2Type<MAPTO>,Int2Type<EDGE>,
	Field<this_type, VOLUME, TR> const & f, index_type s)const->decltype(f[s][0]*1.0)
	{

		auto n = topology_type::_C(topology_type::_Dual(s));
		auto X = topology_type::_D(topology_type::_Dual(s));
		auto Y = topology_type::_R(X);
		auto Z = topology_type::_RR(X);
		return (

		(
				f[(s-Y) - Z][n]*geometry_type::InvVolume((s-Y) - Z) +

				f[(s-Y) + Z][n]*geometry_type::InvVolume((s-Y) + Z)+

				f[(s+Y) - Z][n]*geometry_type::InvVolume((s+Y) - Z)+

				f[(s+Y) + Z][n]*geometry_type::InvVolume((s+Y) + Z)

		)*0.25*geometry_type::Volume(s)

		);
	}
};
template<typename TTopology, template<typename > class TGeo> inline std::ostream &
operator<<(std::ostream & os, RectMesh<TTopology, TGeo> const & d)
{
	d.Print(os);

	return os;
}
}
// namespace simpla

#endif /* MESH_RECTANGLE_H_ */

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
#include "../physics/physical_constants.h"
#include "../utilities/memory_pool.h"
#include "../utilities/type_utilites.h"

namespace simpla
{
template<typename > class EuclideanGeometry;

class OcForest;
/**
 *  Grid is mapped as a rectangle/hexahedrom region;
 *
 */
template<typename TTopology = OcForest, template<typename > class Geometry = EuclideanGeometry>
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
	RectMesh(Args const &... args) :
			geometry_type(static_cast<TTopology const &>(*this)), dt_(1.0), time_(0.0)
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

	coordinates_type GetCoordinates(index_type s) const
	{
		return geometry_type::CoordinatesLocalToGlobal(topology_type::GetCoordinates(s));
	}

	//***************************************************************************************************
	//*	Miscellaneous
	//***************************************************************************************************

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

	//***************************************************************************************************

	coordinates_type CoordinatesLocalToGlobal(index_type s, coordinates_type x) const
	{
		return geometry_type::CoordinatesLocalToGlobal(topology_type::CoordinatesLocalToGlobal(s,x));
	}

	index_type GetIndex(coordinates_type x)const
	{
		return topology_type::GetIndex(geometry_type::CoordinatesGlobalToLocal(x));
	}

	template<typename TF>
	inline typename TF::value_type
	Gather_(TF const &f,coordinates_type r,typename topology_type::compact_index_type shift,unsigned long h=0 ) const
	{
		auto X = (topology_type::_DI >> (h+1));
		auto Y = (topology_type::_DJ >> (h+1));
		auto Z = (topology_type::_DK >> (h+1));

		auto s= topology_type:: CoordinatesGlobalToLocal(&r,shift,h) + (topology_type::_DA >> (h + 1));

		return

		f[((s + X) + Y) + Z]*geometry_type::InvVolume(((s + X) + Y) + Z) * (r[0])* (r[1])* (r[2])+
		f[((s + X) + Y) - Z]*geometry_type::InvVolume(((s + X) + Y) - Z) * (r[0])* (r[1])* (1.0-r[2])+
		f[((s + X) - Y) + Z]*geometry_type::InvVolume(((s + X) - Y) + Z) * (r[0])* (1.0-r[1])* (r[2])+
		f[((s + X) - Y) - Z]*geometry_type::InvVolume(((s + X) - Y) - Z) * (r[0])* (1.0-r[1])* (1.0-r[2])+
		f[((s - X) + Y) + Z]*geometry_type::InvVolume(((s - X) + Y) + Z) * (1.0-r[0])* (r[1])* (r[2])+
		f[((s - X) + Y) - Z]*geometry_type::InvVolume(((s - X) + Y) - Z) * (1.0-r[0])* (r[1])* (1.0-r[2])+
		f[((s - X) - Y) + Z]*geometry_type::InvVolume(((s - X) - Y) + Z) * (1.0-r[0])* (1.0-r[1])* (r[2])+
		f[((s - X) - Y) - Z]*geometry_type::InvVolume(((s - X) - Y) - Z) * (1.0-r[0])* (1.0-r[1])* (1.0-r[2])
		;
	}

	template<typename TExpr>
	inline typename Field<this_type,VERTEX,TExpr>::field_value_type
	Gather(Field<this_type,VERTEX,TExpr>const &f,coordinates_type const &x,unsigned long h=0 ) const
	{
		return Gather_(f,geometry_type::CoordinatesGlobalToLocal(x),0UL,h);
	}

	template<typename TExpr>
	inline typename Field<this_type,EDGE,TExpr>::field_value_type
	Gather(Field<this_type,EDGE,TExpr>const &f,coordinates_type const &y,unsigned long h=0 ) const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);

		return typename Field<this_type,EDGE,TExpr>::field_value_type (
		{
			Gather_(f,x,topology_type::_DI ,h),

			Gather_(f,x,topology_type::_DJ ,h),

			Gather_(f,x,topology_type::_DK ,h)
		});

	}
	template<typename TExpr>
	inline typename Field<this_type,FACE,TExpr>::field_value_type
	Gather(Field<this_type,FACE,TExpr>const &f,coordinates_type y,unsigned long h=0 ) const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);

		return typename Field<this_type,EDGE,TExpr>::field_value_type (
		{
			Gather_(f,x,(topology_type::_DJ|topology_type::_DK) ,h),

			Gather_(f,x,(topology_type::_DK|topology_type::_DI) ,h),

			Gather_(f,x,(topology_type::_DI|topology_type::_DJ) ,h)
		});

	}
	template<typename TExpr>
	inline typename Field<this_type,VOLUME,TExpr>::field_value_type
	Gather(Field<this_type,VOLUME,TExpr>const &f,coordinates_type y,unsigned long h=0 ) const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);
		return Gather_(f,x,topology_type::_DA ,h);

	}

	template<typename TF>
	inline void
	Scatter_( coordinates_type r,typename TF::value_type const & v,typename topology_type::compact_index_type shift,TF *f,unsigned long h =0 ) const
	{
		auto X = (topology_type::_DI >> (h+1));
		auto Y = (topology_type::_DJ >> (h+1));
		auto Z = (topology_type::_DK >> (h+1));

		auto s= topology_type:: CoordinatesGlobalToLocal(&r,shift,h)+ (topology_type::_DA >> (h + 1));

		f->get(((s + X) + Y) + Z)+=v*geometry_type::Volume(((s + X) + Y) + Z) * (r[0])* (r[1])* (r[2]);
		f->get(((s + X) + Y) - Z)+=v*geometry_type::Volume(((s + X) + Y) - Z) * (r[0])* (r[1])* (1.0-r[2]);
		f->get(((s + X) - Y) + Z)+=v*geometry_type::Volume(((s + X) - Y) + Z) * (r[0])* (1.0-r[1])* (r[2]);
		f->get(((s + X) - Y) - Z)+=v*geometry_type::Volume(((s + X) - Y) - Z) * (r[0])* (1.0-r[1])* (1.0-r[2]);
		f->get(((s - X) + Y) + Z)+=v*geometry_type::Volume(((s - X) + Y) + Z) * (1.0-r[0])* (r[1])* (r[2]);
		f->get(((s - X) + Y) - Z)+=v*geometry_type::Volume(((s - X) + Y) - Z) * (1.0-r[0])* (r[1])* (1.0-r[2]);
		f->get(((s - X) - Y) + Z)+=v*geometry_type::Volume(((s - X) - Y) + Z) * (1.0-r[0])* (1.0-r[1])* (r[2]);
		f->get(((s - X) - Y) - Z)+=v*geometry_type::Volume(((s - X) - Y) - Z) * (1.0-r[0])* (1.0-r[1])* (1.0-r[2]);
	}

	template< typename TExpr>
	inline void Scatter( coordinates_type const &y,
	typename Field<this_type,VERTEX,TExpr>::field_value_type const & v ,Field<this_type,VERTEX,TExpr> *f,unsigned long h =0 )const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);

		Scatter_( x,v,0UL,f,h);
	}

	template< typename TExpr>
	inline void Scatter( coordinates_type const &y,
	typename Field<this_type,EDGE,TExpr>::field_value_type const & v ,Field<this_type,EDGE,TExpr> *f,unsigned long h =0 )const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);

		Scatter_( x,v[0],topology_type::_DI,f,h);

		Scatter_( x,v[1],topology_type::_DJ,f,h);

		Scatter_( x,v[2],topology_type::_DK,f,h);
	}

	template< typename TExpr>
	inline void Scatter( coordinates_type const &y,
	typename Field<this_type,FACE,TExpr>::field_value_type const & v ,Field<this_type,FACE,TExpr> *f,unsigned long h =0 )const
	{
		auto x=geometry_type::CoordinatesGlobalToLocal(y);

		Scatter_( x,v[0],(topology_type::_DJ|topology_type::_DK),f,h);

		Scatter_( x,v[1],(topology_type::_DK|topology_type::_DI),f,h);

		Scatter_( x,v[2],(topology_type::_DI|topology_type::_DJ),f,h);
	}

	template< typename TExpr>
	inline void Scatter( coordinates_type const &y ,
	typename Field<this_type,VOLUME,TExpr>::field_value_type const & v ,Field<this_type,VOLUME,TExpr> *f,unsigned long h =0 )const
	{
		auto x = geometry_type::CoordinatesGlobalToLocal(y);

		Scatter_( x,v,topology_type::_DA,f,h);
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

#endif /* MESH_RECTANGLE_H_ */

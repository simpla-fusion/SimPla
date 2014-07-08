/*
 * mesh_rectangle.h
 *
 *  Created on: 2014-2-26
 *      Author: salmon
 */

#ifndef MESH_RECTANGLE_H_
#define MESH_RECTANGLE_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../utilities/sp_type_traits.h"
#include "../utilities/container_dense.h"
#include "../physics/constants.h"

#include "interpolator.h"

namespace simpla
{

/**
 *   \defgroup  Mesh Mesh
 *
 *     define all basic discrete scheme, i.e. finite difference , finite volume, finite element
 *
 *
 *   @{
 *      \defgroup  Geometry Geometry
 *        \brief coordinates and metric dependent information
 *     	\defgroup  Topology Topology
 *        \brief coordinates and metric  independent information
 *   @}
 */

template<typename, unsigned int, typename > class Field;

template<typename TM, typename Policy> class Interpolator;

/**
 *  \brief template of Mesh
 */
template<typename TGeometry, bool EnableSpectralMethod = false>
class Mesh: public TGeometry
{
public:
	typedef TGeometry geometry_type;

	static constexpr bool enable_spectral_method = EnableSpectralMethod;

	typedef Mesh<geometry_type, enable_spectral_method> this_type;

	typedef typename std::conditional<enable_spectral_method, Complex, Real>::type scalar_type;

	typedef typename geometry_type::topology_type topology_type;

	typedef typename geometry_type::coordinates_type coordinates_type;

	typedef typename geometry_type::index_type index_type;

	typedef typename geometry_type::compact_index_type compact_index_type;

	typedef Interpolator<this_type, std::nullptr_t> interpolator_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

	nTuple<NDIMS, scalar_type> k_imag = { 0, 0, 0 };

	Mesh()
			: geometry_type()
	{
	}

	template<typename ... Args>
	Mesh(Args && ... args)
			: geometry_type(std::forward<Args>(args)...)
	{
		UpdateK(&k_imag);
	}

	~Mesh()
	{
	}

	Mesh(const this_type&) = delete;

	this_type & operator=(const this_type&) = delete;

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}

	template<typename OS>
	OS & Print(OS &os) const
	{
		geometry_type::Print(os) << std::endl

		<< " , " << "K_img = " << k_imag;

		return os;
	}

	template<typename ...Args>
	inline void set_extents(Args&& ... args)
	{
		geometry_type::set_extents(std::forward<Args>(args)...);

		UpdateK(&k_imag);
	}

	//******************************************************************************************************

private:
	template<typename T>
	void UpdateK(T* k)
	{
		auto dims = geometry_type::get_dimensions();
		auto extents = geometry_type::get_extents();
		coordinates_type xmin = std::get<0>(extents);
		coordinates_type xmax = std::get<1>(extents);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] - xmin[i]) < EPSILON)
			{
				xmax[i] = xmin[i];
			}
		}
		geometry_type::set_extents(xmin, xmax);

	}
	void UpdateK(nTuple<NDIMS, Complex>* k)
	{
		auto dims = geometry_type::get_dimensions();
		auto extents = geometry_type::get_extents();
		auto xmin = std::get<0>(extents);
		auto xmax = std::get<1>(extents);

		for (int i = 0; i < NDIMS; ++i)
		{
			(*k)[i] = 0;

			if (dims[i] <= 1)
			{
				if (xmax[i] > xmin[i])
					(*k)[i] = Complex(0, TWOPI / (xmax[i] - xmin[i]));

				xmax[i] = xmin[i];
			}
		}
		geometry_type::set_extents(xmin, xmax);

	}
public:

	template<typename TV> using DefaultContainer=DenseContainer<compact_index_type,TV>;

	template<unsigned int IFORM, typename TV> using field=Field<this_type,IFORM,DefaultContainer<TV>>;

	template<typename TF, typename ... Args> TF  //
	make_field(typename topology_type::range_type range, Args && ... args) const
	{
		return std::move(TF(*this, range, topology_type::make_hash(range), std::forward<Args>(args)...));
	}

	template<typename TF, typename ... Args> inline auto //
	make_field(Args && ... args) const
	DECL_RET_TYPE((make_field<TF>(topology_type::Select(TF::IForm),std::forward<Args>(args)...)))

	template<unsigned int IFORM, typename TV, typename ...Args> inline auto //
	make_field(Args &&...args) const
	DECL_RET_TYPE((make_field<field< IFORM, TV >>(std::forward<Args>(args)... )))

	//******************************************************************************************************

	static constexpr unsigned int get_num_of_dimensions()
	{
		return NDIMS;
	}
	Real CheckCourantDt(nTuple<3, Real> const & u) const
	{

		Real dt = geometry_type::get_dt();

		auto dims = geometry_type::get_dimensions();
		auto extent = geometry_type::get_extents();

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
		return CheckCourantDt(nTuple<3, Real>( { speed, speed, speed }));
	}

	template<typename ...Args>
	inline auto Gather(Args && ... args) const
	DECL_RET_TYPE(interpolator_type::Gather(*this,std::forward<Args>(args)...))

	template<typename ...Args>
	inline void Scatter(Args && ... args) const
	{
		interpolator_type::Scatter(*this, std::forward<Args>(args)...);
	}

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>, Field<this_type, VERTEX, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = topology_type::DeltaIndex(s);

		return

		(get_value(f, s + D) * geometry_type::Volume(s + D) - get_value(f, s - D) * geometry_type::Volume(s - D))
		        * geometry_type::InvVolume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        + get_value(f, s + D) * k_imag[geometry_type::ComponentNum(D)];
#endif
		;
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>, Field<this_type, EDGE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DeltaIndex(topology_type::Dual(s));
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		return (

		(get_value(f, s + Y) * geometry_type::Volume(s + Y) //
		- get_value(f, s - Y) * geometry_type::Volume(s - Y)) //
		- (get_value(f, s + Z) * geometry_type::Volume(s + Z) //
		- get_value(f, s - Z) * geometry_type::Volume(s - Z)) //

		) * geometry_type::InvVolume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        + get_value(f, s + Y) * k_imag[geometry_type::ComponentNum(Y)]
		        - get_value(f, s + Z) * k_imag[geometry_type::ComponentNum(Z)]
#endif

		;
	}

	template<typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>, Field<this_type, FACE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return (

		get_value(f, s + X) * geometry_type::Volume(s + X)

		- get_value(f, s - X) * geometry_type::Volume(s - X) //
		+ get_value(f, s + Y) * geometry_type::Volume(s + Y) //
		- get_value(f, s - Y) * geometry_type::Volume(s - Y) //
		+ get_value(f, s + Z) * geometry_type::Volume(s + Z) //
		- get_value(f, s - Z) * geometry_type::Volume(s - Z) //

		) * geometry_type::InvVolume(s)

#ifndef DISABLE_SPECTRAL_METHD

		        + get_value(f, s + X) * k_imag[geometry_type::ComponentNum(X)]
		        + get_value(f, s + Y) * k_imag[geometry_type::ComponentNum(Y)]
		        + get_value(f, s + Z) * k_imag[geometry_type::ComponentNum(Z)]

#endif

		;
	}

	template<unsigned int IL, typename TL> void OpEval(Int2Type<EXTRIORDERIVATIVE>, Field<this_type, IL, TL> const & f,
	        compact_index_type s) const = delete;

	template<unsigned int IL, typename TL> void OpEval(Int2Type<CODIFFERENTIAL>, Field<this_type, IL, TL> const & f,
	        compact_index_type s) const = delete;

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>, Field<this_type, EDGE, TL> const & f,
	        compact_index_type s) const->decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		-(

		get_value(f, s + X) * geometry_type::DualVolume(s + X)

		- get_value(f, s - X) * geometry_type::DualVolume(s - X)

		+ get_value(f, s + Y) * geometry_type::DualVolume(s + Y)

		- get_value(f, s - Y) * geometry_type::DualVolume(s - Y)

		+ get_value(f, s + Z) * geometry_type::DualVolume(s + Z)

		- get_value(f, s - Z) * geometry_type::DualVolume(s - Z)

		) * geometry_type::InvDualVolume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + X) * k_imag[geometry_type::ComponentNum(X)]
		        - get_value(f, s + Y) * k_imag[geometry_type::ComponentNum(Y)]
		        - get_value(f, s + Z) * k_imag[geometry_type::ComponentNum(Z)]
#endif
		;

	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>, Field<this_type, FACE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DeltaIndex(s);
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		return

		-(

		(get_value(f, s + Y) * (geometry_type::DualVolume(s + Y))
		        - get_value(f, s - Y) * (geometry_type::DualVolume(s - Y)))

		        - (get_value(f, s + Z) * (geometry_type::DualVolume(s + Z))
		                - get_value(f, s - Z) * (geometry_type::DualVolume(s - Z)))

		) * geometry_type::InvDualVolume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + Y) * k_imag[geometry_type::ComponentNum(Y)]
		        + get_value(f, s + Z) * k_imag[geometry_type::ComponentNum(Z)]
#endif
		;
	}

	template<typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>, Field<this_type, VOLUME, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = topology_type::DeltaIndex(topology_type::Dual(s));
		return

		-(

		get_value(f, s + D) * (geometry_type::DualVolume(s + D)) //
		- get_value(f, s - D) * (geometry_type::DualVolume(s - D))

		) * geometry_type::InvDualVolume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + D) * k_imag[geometry_type::ComponentNum(D)]
#endif

		;
	}

//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, VERTEX, TL> const &l,
	        Field<this_type, VERTEX, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		return get_value(l, s) * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, VERTEX, TL> const &l,
	        Field<this_type, EDGE, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DeltaIndex(s);

		return (get_value(l, s - X) + get_value(l, s + X)) * 0.5 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, VERTEX, TL> const &l,
	        Field<this_type, FACE, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DeltaIndex(topology_type::Dual(s));
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		return (

		get_value(l, (s - Y) - Z) +

		get_value(l, (s - Y) + Z) +

		get_value(l, (s + Y) - Z) +

		get_value(l, (s + Y) + Z)

		) * 0.25 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, VERTEX, TL> const &l,
	        Field<this_type, VOLUME, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return (

		get_value(l, ((s - X) - Y) - Z) +

		get_value(l, ((s - X) - Y) + Z) +

		get_value(l, ((s - X) + Y) - Z) +

		get_value(l, ((s - X) + Y) + Z) +

		get_value(l, ((s + X) - Y) - Z) +

		get_value(l, ((s + X) - Y) + Z) +

		get_value(l, ((s + X) + Y) - Z) +

		get_value(l, ((s + X) + Y) + Z)

		) * 0.125 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, EDGE, TL> const &l,
	        Field<this_type, VERTEX, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DeltaIndex(s);
		return get_value(l, s) * (get_value(r, s - X) + get_value(r, s + X)) * 0.5;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, EDGE, TL> const &l,
	        Field<this_type, EDGE, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::DeltaIndex(topology_type::Roate(topology_type::Dual(s)));
		auto Z = topology_type::DeltaIndex(topology_type::InverseRoate(topology_type::Dual(s)));

		return ((get_value(l, s - Y) + get_value(l, s + Y)) * (get_value(l, s - Z) + get_value(l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, EDGE, TL> const &l,
	        Field<this_type, FACE, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		(

		(get_value(l, (s - Y) - Z) + get_value(l, (s - Y) + Z) + get_value(l, (s + Y) - Z) + get_value(l, (s + Y) + Z))
		        * (get_value(r, s - X) + get_value(r, s + X))
		        +

		        (get_value(l, (s - Z) - X) + get_value(l, (s - Z) + X) + get_value(l, (s + Z) - X)
		                + get_value(l, (s + Z) + X)) * (get_value(r, s - Y) + get_value(r, s + Y))
		        +

		        (get_value(l, (s - X) - Y) + get_value(l, (s - X) + Y) + get_value(l, (s + X) - Y)
		                + get_value(l, (s + X) + Y)) * (get_value(r, s - Z) + get_value(r, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, FACE, TL> const &l,
	        Field<this_type, VERTEX, TR> const &r, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::DeltaIndex(topology_type::Roate(topology_type::Dual(s)));
		auto Z = topology_type::DeltaIndex(topology_type::InverseRoate(topology_type::Dual(s)));

		return get_value(l, s)
		        * (get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z) + get_value(r, (s + Y) - Z)
		                + get_value(r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, FACE, TL> const &r,
	        Field<this_type, EDGE, TR> const &l, compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		(

		(get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z) + get_value(r, (s + Y) - Z) + get_value(r, (s + Y) + Z))
		        * (get_value(l, s - X) + get_value(l, s + X))

		        + (get_value(r, (s - Z) - X) + get_value(r, (s - Z) + X) + get_value(r, (s + Z) - X)
		                + get_value(r, (s + Z) + X)) * (get_value(l, s - Y) + get_value(l, s + Y))

		        + (get_value(r, (s - X) - Y) + get_value(r, (s - X) + Y) + get_value(r, (s + X) - Y)
		                + get_value(r, (s + X) + Y)) * (get_value(l, s - Z) + get_value(l, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<WEDGE>, Field<this_type, VOLUME, TL> const &l,
	        Field<this_type, VERTEX, TR> const &r, compact_index_type s) const ->decltype(get_value(r,s)*get_value(l,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		get_value(l, s) * (

		get_value(r, ((s - X) - Y) - Z) + //
		        get_value(r, ((s - X) - Y) + Z) + //
		        get_value(r, ((s - X) + Y) - Z) + //
		        get_value(r, ((s - X) + Y) + Z) + //
		        get_value(r, ((s + X) - Y) - Z) + //
		        get_value(r, ((s + X) - Y) + Z) + //
		        get_value(r, ((s + X) + Y) - Z) + //
		        get_value(r, ((s + X) + Y) + Z)   //

		) * 0.125;
	}

//***************************************************************************************************

	template<unsigned int IL, typename TL> inline auto OpEval(Int2Type<HODGESTAR>, Field<this_type, IL, TL> const & f,
	        compact_index_type s) const-> typename std::remove_reference<decltype(get_value(f,s))>::type
	{
//		auto X = topology_type::DI(0,s);
//		auto Y = topology_type::DI(1,s);
//		auto Z =topology_type::DI(2,s);
//
//		return
//
//		(
//
//		get_value(f,((s + X) - Y) - Z)*geometry_type::InvVolume(((s + X) - Y) - Z) +
//
//		get_value(f,((s + X) - Y) + Z)*geometry_type::InvVolume(((s + X) - Y) + Z) +
//
//		get_value(f,((s + X) + Y) - Z)*geometry_type::InvVolume(((s + X) + Y) - Z) +
//
//		get_value(f,((s + X) + Y) + Z)*geometry_type::InvVolume(((s + X) + Y) + Z) +
//
//		get_value(f,((s - X) - Y) - Z)*geometry_type::InvVolume(((s - X) - Y) - Z) +
//
//		get_value(f,((s - X) - Y) + Z)*geometry_type::InvVolume(((s - X) - Y) + Z) +
//
//		get_value(f,((s - X) + Y) - Z)*geometry_type::InvVolume(((s - X) + Y) - Z) +
//
//		get_value(f,((s - X) + Y) + Z)*geometry_type::InvVolume(((s - X) + Y) + Z)
//
//		) * 0.125 * geometry_type::Volume(s);

		return get_value(f, s) /** geometry_type::HodgeStarVolumeScale(s)*/;
	}

	template<typename TL, typename TR> void OpEval(Int2Type<INTERIOR_PRODUCT>, nTuple<NDIMS, TR> const & v,
	        Field<this_type, VERTEX, TL> const & f, compact_index_type s) const = delete;

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>, nTuple<NDIMS, TR> const & v,
	        Field<this_type, EDGE, TL> const & f, compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		(get_value(f, s + X) - get_value(f, s - X)) * 0.5 * v[0]  //
		+ (get_value(f, s + Y) - get_value(f, s - Y)) * 0.5 * v[1]   //
		+ (get_value(f, s + Z) - get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>, nTuple<NDIMS, TR> const & v,
	        Field<this_type, FACE, TL> const & f, compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::ComponentNum(s);

		auto X = topology_type::DeltaIndex(s);
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);
		return

		(get_value(f, s + Y) + get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(get_value(f, s + Z) + get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> inline auto OpEval(Int2Type<INTERIOR_PRODUCT>, nTuple<NDIMS, TR> const & v,
	        Field<this_type, VOLUME, TL> const & f, compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::ComponentNum(topology_type::Dual(s));
		unsigned int D = topology_type::DeltaIndex(topology_type::Dual(s));

		return (get_value(f, s + D) - get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<unsigned int N, typename TL> inline auto OpEval(Int2Type<EXTRIORDERIVATIVE>,
	        Field<this_type, EDGE, TL> const & f, Int2Type<N>,
	        compact_index_type s) const-> decltype(get_value(f,s)-get_value(f,s))
	{

		auto X = topology_type::DeltaIndex(topology_type::Dual(s));
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		Y = (topology_type::ComponentNum(Y) == N) ? Y : 0UL;
		Z = (topology_type::ComponentNum(Z) == N) ? Z : 0UL;

		return (get_value(f, s + Y) - get_value(f, s - Y)) - (get_value(f, s + Z) - get_value(f, s - Z));
	}

	template<unsigned int N, typename TL> inline auto OpEval(Int2Type<CODIFFERENTIAL>,
	        Field<this_type, FACE, TL> const & f, Int2Type<N>,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{

		auto X = topology_type::DeltaIndex(s);
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		Y = (topology_type::ComponentNum(Y) == N) ? Y : 0UL;
		Z = (topology_type::ComponentNum(Z) == N) ? Z : 0UL;

		return (

		get_value(f, s + Y) * (geometry_type::DualVolume(s + Y))      //
		- get_value(f, s - Y) * (geometry_type::DualVolume(s - Y))    //
		- get_value(f, s + Z) * (geometry_type::DualVolume(s + Z))    //
		+ get_value(f, s - Z) * (geometry_type::DualVolume(s - Z))    //

		) * geometry_type::InvDualVolume(s);
	}
	template<unsigned int IL, typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<IL> const &,
	        Field<this_type, IL, TR> const & f, compact_index_type s) const
	        DECL_RET_TYPE(get_value(f,s))

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<VERTEX> const &,
	        Field<this_type, EDGE, TR> const & f,
	        compact_index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3, typename std::remove_reference<decltype(get_value(f,s))>::type>( {

		(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
		(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
		(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<EDGE> const &,
	        Field<this_type, VERTEX, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::ComponentNum(s);
		auto D = topology_type::DeltaIndex(s);

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<VERTEX> const &,
	        Field<this_type, FACE, TR> const & f,
	        compact_index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3, typename std::remove_reference<decltype(get_value(f,s))>::type>( { (

		get_value(f, (s - Y) - Z) +

		get_value(f, (s - Y) + Z) +

		get_value(f, (s + Y) - Z) +

		get_value(f, (s + Y) + Z)

		) * 0.25,

		(

		get_value(f, (s - Z) - X) +

		get_value(f, (s - Z) + X) +

		get_value(f, (s + Z) - X) +

		get_value(f, (s + Z) + X)

		) * 0.25,

		(

		get_value(f, (s - X) - Y) +

		get_value(f, (s - X) + Y) +

		get_value(f, (s + X) - Y) +

		get_value(f, (s + X) + Y)

		) * 0.25

		});
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<FACE> const &,
	        Field<this_type, VERTEX, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::ComponentNum(topology_type::Dual(s));
		auto X = topology_type::DeltaIndex(topology_type::Dual(s));
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);

		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<VOLUME>, Field<this_type, FACE, TR> const & f,
	        compact_index_type s) const->nTuple<3,decltype(get_value(f,s) )>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3, typename std::remove_reference<decltype(get_value(f,s))>::type>( {

		(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
		(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
		(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<FACE>, Field<this_type, VOLUME, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::ComponentNum(topology_type::Dual(s));
		auto D = topology_type::DeltaIndex(topology_type::Dual(s));

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<VOLUME>, Field<this_type, EDGE, TR> const & f,
	        compact_index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s) )>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3, typename std::remove_reference<decltype(get_value(f,s))>::type>( { (

		get_value(f, (s - Y) - Z) +

		get_value(f, (s - Y) + Z) +

		get_value(f, (s + Y) - Z) +

		get_value(f, (s + Y) + Z)

		) * 0.25,

		(

		get_value(f, (s - Z) - X) +

		get_value(f, (s - Z) + X) +

		get_value(f, (s + Z) - X) +

		get_value(f, (s + Z) + X)

		) * 0.25,

		(

		get_value(f, (s - X) - Y) +

		get_value(f, (s - X) + Y) +

		get_value(f, (s + X) - Y) +

		get_value(f, (s + X) + Y)

		) * 0.25,

		});
	}

	template<typename TR> inline auto OpEval(Int2Type<MAPTO>, Int2Type<EDGE>, Field<this_type, VOLUME, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::ComponentNum(topology_type::Dual(s));
		auto X = topology_type::DeltaIndex(topology_type::Dual(s));
		auto Y = topology_type::Roate(X);
		auto Z = topology_type::InverseRoate(X);
		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}
};

}
// namespace simpla

#endif /* MESH_RECTANGLE_H_ */

/*
 * mesh_rectangle.h
 *
 *  created on: 2014-2-26
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

//	typedef Interpolator<this_type, std::nullptr_t> interpolator_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

	nTuple<NDIMS, scalar_type> k_imag/* = { 0, 0, 0 }*/;

	Mesh()
	{
	}

	~Mesh()
	{
	}

	Mesh(const this_type&) = delete;

	template<typename TDict>
	bool load(TDict const& dict)
	{
		return geometry_type::load(dict);
	}

	this_type & operator=(const this_type&) = delete;

	inline bool operator==(this_type const & r) const
	{
		return (this == &r);
	}
	static std::string get_type_as_string_static()
	{
		return geometry_type::get_type_as_string_static() + "_" + topology_type::get_type_as_string_static()
		        + ((enable_spectral_method) ? "_kz" : "");
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}
	template<typename OS>
	OS & print(OS &os) const
	{
		geometry_type::print(os);

		if (enable_spectral_method)
		{
			os << " K_img = " << k_imag << std::endl;
		}

		return os;
	}

	template<typename ...Args>
	inline void set_extents(Args&& ... args)
	{
		geometry_type::set_extents(std::forward<Args>(args)...);
	}

	void update()
	{
		updateK(&k_imag);
		geometry_type::update();
		geometry_type::updatedt(k_imag);
	}
private:
	bool is_ready_ = false;
public:

	bool is_ready() const
	{
		return is_ready_;
	}

	//******************************************************************************************************

private:
	template<typename T>
	bool updateK(T* k)
	{
		return true;
	}
	bool updateK(nTuple<NDIMS, Complex>* k)
	{

		coordinates_type xmin, xmax;

		auto dims = geometry_type::get_dimensions();

		std::tie(xmin, xmax) = geometry_type::get_extents();

		for (int i = 0; i < NDIMS; ++i)
		{
			(*k)[i] = 0;

			if (dims[i] <= 1 && xmax[i] > xmin[i])
			{
				(*k)[i] = Complex(0, TWOPI / (xmax[i] - xmin[i]));
			}
		}
		return true;
	}
public:

	template<typename TV> using DefaultContainer=DenseContainer<compact_index_type,TV>;

	template<unsigned int IFORM, typename TV> using field=Field<this_type,IFORM,DefaultContainer<TV>>;

//	template<typename TF, typename ... Args> TF  //
//	make_field(typename topology_type::range_type range, Args && ... args) const
//	{
//		return std::move(
//		        TF(*this, range, topology_type::max_hash_value(range), topology_type::make_hash(range),
//		                std::forward<Args>(args)...));
//	}
//
//	template<typename TF, typename ... Args> inline auto //
//	make_field(Args && ... args) const
//	DECL_RET_TYPE((make_field<TF>(topology_type::select(TF::IForm),std::forward<Args>(args)...)))
//
//	template<unsigned int IFORM, typename TV, typename ...Args> inline auto //
//	make_field(Args &&...args) const
//	DECL_RET_TYPE((make_field<field< IFORM, TV >>(std::forward<Args>(args)... )))

	template<typename TF> TF  //
	make_field(typename topology_type::range_type range) const
	{
		return std::move(TF(*this, range, topology_type::max_hash_value(range), topology_type::make_hash(range)));
	}

	template<typename TF> inline TF //
	make_field() const
	{
		auto range = topology_type::select(TF::IForm);
		return std::move(TF(*this, range, topology_type::max_hash_value(range), topology_type::make_hash(range)));
	}

	template<unsigned int IFORM, typename TV> inline field<IFORM, TV> //
	make_field() const
	{
		auto range = topology_type::select(IFORM);
		return std::move(
		        field<IFORM, TV>(*this, range, topology_type::max_hash_value(range), topology_type::make_hash(range)));
	}

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

//	template<typename ...Args>
//	inline auto Gather(Args && ... args) const
//	DECL_RET_TYPE(interpolator_type::Gather(*this,std::forward<Args>(args)...))
//
//	template<typename ...Args>
//	inline void Scatter(Args && ... args) const
//	{
//		interpolator_type::Scatter(*this, std::forward<Args>(args)...);
//	}

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
	        Field<this_type, VERTEX, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = topology_type::delta_index(s);

		return

		(get_value(f, s + D) * geometry_type::volume(s + D) - get_value(f, s - D) * geometry_type::volume(s - D))
		        * geometry_type::inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        + get_value(f, s + D) * k_imag[geometry_type::component_number(D)];
#endif
		;
	}

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
	        Field<this_type, EDGE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		(get_value(f, s + Y) * geometry_type::volume(s + Y) //
		- get_value(f, s - Y) * geometry_type::volume(s - Y)) //
		- (get_value(f, s + Z) * geometry_type::volume(s + Z) //
		- get_value(f, s - Z) * geometry_type::volume(s - Z)) //

		) * geometry_type::inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        + get_value(f, s + Y) * k_imag[geometry_type::component_number(Y)]
		        - get_value(f, s + Z) * k_imag[geometry_type::component_number(Z)]
#endif

		;
	}

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
	        Field<this_type, FACE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return (

		get_value(f, s + X) * geometry_type::volume(s + X)

		- get_value(f, s - X) * geometry_type::volume(s - X) //
		+ get_value(f, s + Y) * geometry_type::volume(s + Y) //
		- get_value(f, s - Y) * geometry_type::volume(s - Y) //
		+ get_value(f, s + Z) * geometry_type::volume(s + Z) //
		- get_value(f, s - Z) * geometry_type::volume(s - Z) //

		) * geometry_type::inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD

		        + get_value(f, s + X) * k_imag[geometry_type::component_number(X)]
		        + get_value(f, s + Y) * k_imag[geometry_type::component_number(Y)]
		        + get_value(f, s + Z) * k_imag[geometry_type::component_number(Z)]

#endif

		;
	}

	template<unsigned int IL, typename TL> void OpEval(std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
	        Field<this_type, IL, TL> const & f, compact_index_type s) const = delete;

	template<unsigned int IL, typename TL> void OpEval(std::integral_constant<unsigned int, CODIFFERENTIAL>,
	        Field<this_type, IL, TL> const & f, compact_index_type s) const = delete;

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, CODIFFERENTIAL>,
	        Field<this_type, EDGE, TL> const & f,
	        compact_index_type s) const->decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		-(

		get_value(f, s + X) * geometry_type::dual_volume(s + X)

		- get_value(f, s - X) * geometry_type::dual_volume(s - X)

		+ get_value(f, s + Y) * geometry_type::dual_volume(s + Y)

		- get_value(f, s - Y) * geometry_type::dual_volume(s - Y)

		+ get_value(f, s + Z) * geometry_type::dual_volume(s + Z)

		- get_value(f, s - Z) * geometry_type::dual_volume(s - Z)

		) * geometry_type::inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + X) * k_imag[geometry_type::component_number(X)]
		        - get_value(f, s + Y) * k_imag[geometry_type::component_number(Y)]
		        - get_value(f, s + Z) * k_imag[geometry_type::component_number(Z)]
#endif
		;

	}

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, CODIFFERENTIAL>,
	        Field<this_type, FACE, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return

		-(

		(get_value(f, s + Y) * (geometry_type::dual_volume(s + Y))
		        - get_value(f, s - Y) * (geometry_type::dual_volume(s - Y)))

		        - (get_value(f, s + Z) * (geometry_type::dual_volume(s + Z))
		                - get_value(f, s - Z) * (geometry_type::dual_volume(s - Z)))

		) * geometry_type::inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + Y) * k_imag[geometry_type::component_number(Y)]
		        + get_value(f, s + Z) * k_imag[geometry_type::component_number(Z)]
#endif
		;
	}

	template<typename TL> inline auto OpEval(std::integral_constant<unsigned int, CODIFFERENTIAL>,
	        Field<this_type, VOLUME, TL> const & f,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = topology_type::delta_index(topology_type::dual(s));
		return

		-(

		get_value(f, s + D) * (geometry_type::dual_volume(s + D)) //
		- get_value(f, s - D) * (geometry_type::dual_volume(s - D))

		) * geometry_type::inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
		        - get_value(f, s + D) * k_imag[geometry_type::component_number(D)]
#endif

		;
	}

//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, VERTEX, TL> const &l, Field<this_type, VERTEX, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		return get_value(l, s) * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, VERTEX, TL> const &l, Field<this_type, EDGE, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(s);

		return (get_value(l, s - X) + get_value(l, s + X)) * 0.5 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, VERTEX, TL> const &l, Field<this_type, FACE, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		get_value(l, (s - Y) - Z) +

		get_value(l, (s - Y) + Z) +

		get_value(l, (s + Y) - Z) +

		get_value(l, (s + Y) + Z)

		) * 0.25 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, VERTEX, TL> const &l, Field<this_type, VOLUME, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, EDGE, TL> const &l, Field<this_type, VERTEX, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(s);
		return get_value(l, s) * (get_value(r, s - X) + get_value(r, s + X)) * 0.5;
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, EDGE, TL> const &l, Field<this_type, EDGE, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::delta_index(topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(topology_type::inverse_roate(topology_type::dual(s)));

		return ((get_value(l, s - Y) + get_value(l, s + Y)) * (get_value(l, s - Z) + get_value(l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, EDGE, TL> const &l, Field<this_type, FACE, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, FACE, TL> const &l, Field<this_type, VERTEX, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::delta_index(topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(topology_type::inverse_roate(topology_type::dual(s)));

		return get_value(l, s)
		        * (get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z) + get_value(r, (s + Y) - Z)
		                + get_value(r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, FACE, TL> const &r, Field<this_type, EDGE, TR> const &l,
	        compact_index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, WEDGE>,
	        Field<this_type, VOLUME, TL> const &l, Field<this_type, VERTEX, TR> const &r,
	        compact_index_type s) const ->decltype(get_value(r,s)*get_value(l,s))
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

	template<unsigned int IL, typename TL> inline auto OpEval(std::integral_constant<unsigned int, HODGESTAR>,
	        Field<this_type, IL, TL> const & f,
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
//		get_value(f,((s + X) - Y) - Z)*geometry_type::inv_volume(((s + X) - Y) - Z) +
//
//		get_value(f,((s + X) - Y) + Z)*geometry_type::inv_volume(((s + X) - Y) + Z) +
//
//		get_value(f,((s + X) + Y) - Z)*geometry_type::inv_volume(((s + X) + Y) - Z) +
//
//		get_value(f,((s + X) + Y) + Z)*geometry_type::inv_volume(((s + X) + Y) + Z) +
//
//		get_value(f,((s - X) - Y) - Z)*geometry_type::inv_volume(((s - X) - Y) - Z) +
//
//		get_value(f,((s - X) - Y) + Z)*geometry_type::inv_volume(((s - X) - Y) + Z) +
//
//		get_value(f,((s - X) + Y) - Z)*geometry_type::inv_volume(((s - X) + Y) - Z) +
//
//		get_value(f,((s - X) + Y) + Z)*geometry_type::inv_volume(((s - X) + Y) + Z)
//
//		) * 0.125 * geometry_type::volume(s);

		return get_value(f, s) /** geometry_type::HodgeStarVolumeScale(s)*/;
	}

	template<typename TL, typename TR> void OpEval(std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
	        nTuple<NDIMS, TR> const & v, Field<this_type, VERTEX, TL> const & f, compact_index_type s) const = delete;

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
	        nTuple<NDIMS, TR> const & v, Field<this_type, EDGE, TL> const & f,
	        compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		(get_value(f, s + X) - get_value(f, s - X)) * 0.5 * v[0]  //
		+ (get_value(f, s + Y) - get_value(f, s - Y)) * 0.5 * v[1]   //
		+ (get_value(f, s + Z) - get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
	        nTuple<NDIMS, TR> const & v, Field<this_type, FACE, TL> const & f,
	        compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::component_number(s);

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
		return

		(get_value(f, s + Y) + get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(get_value(f, s + Z) + get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
	        nTuple<NDIMS, TR> const & v, Field<this_type, VOLUME, TL> const & f,
	        compact_index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::component_number(topology_type::dual(s));
		unsigned int D = topology_type::delta_index(topology_type::dual(s));

		return (get_value(f, s + D) - get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<unsigned int N, typename TL> inline auto OpEval(std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
	        Field<this_type, EDGE, TL> const & f, std::integral_constant<unsigned int, N>,
	        compact_index_type s) const-> decltype(get_value(f,s)-get_value(f,s))
	{

		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (get_value(f, s + Y) - get_value(f, s - Y)) - (get_value(f, s + Z) - get_value(f, s - Z));
	}

	template<unsigned int N, typename TL> inline auto OpEval(std::integral_constant<unsigned int, CODIFFERENTIAL>,
	        Field<this_type, FACE, TL> const & f, std::integral_constant<unsigned int, N>,
	        compact_index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (

		get_value(f, s + Y) * (geometry_type::dual_volume(s + Y))      //
		- get_value(f, s - Y) * (geometry_type::dual_volume(s - Y))    //
		- get_value(f, s + Z) * (geometry_type::dual_volume(s + Z))    //
		+ get_value(f, s - Z) * (geometry_type::dual_volume(s - Z))    //

		) * geometry_type::inv_dual_volume(s);
	}
	template<unsigned int IL, typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, IL> const &, Field<this_type, IL, TR> const & f,
	        compact_index_type s) const
	        DECL_RET_TYPE(get_value(f,s))

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, VERTEX> const &, Field<this_type, EDGE, TR> const & f,
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

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, EDGE> const &, Field<this_type, VERTEX, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(s);
		auto D = topology_type::delta_index(s);

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, VERTEX> const &, Field<this_type, FACE, TR> const & f,
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

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, FACE> const &, Field<this_type, VERTEX, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, VOLUME>, Field<this_type, FACE, TR> const & f,
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

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, FACE>, Field<this_type, VOLUME, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto D = topology_type::delta_index(topology_type::dual(s));

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, VOLUME>, Field<this_type, EDGE, TR> const & f,
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

	template<typename TR> inline auto OpEval(std::integral_constant<unsigned int, MAPTO>,
	        std::integral_constant<unsigned int, EDGE>, Field<this_type, VOLUME, TR> const & f,
	        compact_index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
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

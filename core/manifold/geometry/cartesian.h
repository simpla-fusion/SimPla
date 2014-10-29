/*
 * coordinates_cartesian.h
 *
 *  created on: 2014-3-13
 *      Author: salmon
 */

#ifndef COORDINATES_CARTESIAN_H_
#define COORDINATES_CARTESIAN_H_

#include <stddef.h>
#include <string>
#include <tuple>
#include <type_traits>

#include "../../physics/constants.h"
#include "../../physics/physical_constants.h"
#include "../../utilities/log.h"
#include "../../utilities/primitives.h"
#include "../../utilities/sp_type_traits.h"

namespace simpla
{
/**
 *  \ingroup Geometry
 *
 *  \brief  Cartesian coordiantes (X Y Z)
 *
 */
template<typename TTopology, size_t ZAXIS = CARTESIAN_ZAXIS>
struct CartesianCoordinates: public TTopology
{
private:
	bool is_ready_ = false;
public:
	typedef TTopology topology_type;
	typedef CartesianCoordinates<topology_type> this_type;

	static constexpr size_t ndims = topology_type::ndims;

	static constexpr size_t XAxis = (ZAXIS + 1) % 3;
	static constexpr size_t YAxis = (ZAXIS + 2) % 3;
	static constexpr size_t ZAxis = ZAXIS;

	typedef Real scalar_type;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::iterator iterator;

	CartesianCoordinates(this_type const & rhs) = delete;

	CartesianCoordinates() :
			topology_type()
	{

		xmin_ = coordinates_type(
		{ 0, 0, 0 });

		xmax_ = coordinates_type(
		{ 1, 1, 1 });

		inv_length_ = coordinates_type(
		{ 1.0, 1.0, 1.0 });

		length_ = coordinates_type(
		{ 1.0, 1.0, 1.0 });

		shift_ = coordinates_type(
		{ 0, 0, 0 });
	}

	template<typename ... Args>
	CartesianCoordinates(Args && ... args) :
			topology_type(std::forward<Args>(args)...)
	{
		load(std::forward<Args>(args)...);
	}

	~CartesianCoordinates()
	{
	}

	static std::string get_type_as_string_static()
	{
		return "Cartesian";
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}
	//***************************************************************************************************
	// Geometric properties
	// Metric
	//***************************************************************************************************

	Real dt_ = 0.0;
	Real time0_ = 0.0;
	Real CFL_ = 0.5;

	// Time
	void next_timestep()
	{
		topology_type::next_timestep();
	}
	void set_time(Real p_time)
	{
		time0_ = p_time;
	}
	Real get_time() const
	{
		return static_cast<double>(topology_type::get_clock()) * dt_ + time0_;
	}

	void set_dt(Real p_dt)
	{
		dt_ = p_dt;
	}

	Real get_dt() const
	{
		return dt_;
	}

	bool is_ready() const
	{
		return is_ready_ && topology_type::is_ready();
	}

	coordinates_type xmin_ /*= { 0, 0, 0 }*/;

	coordinates_type xmax_ /*= { 1, 1, 1 }*/;

	coordinates_type inv_length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type shift_/* = { 0, 0, 0 }*/;

	bool update();

	void updatedt(Real dx2 = 0.0)
	{
		DEFINE_PHYSICAL_CONST

		auto dx_ = dx();

		Real safe_dt = CFL_
				* std::sqrt(dx_[0] * dx_[0] + dx_[1] * dx_[1] + dx_[2] * dx_[2])
				/ speed_of_light;

		if (dt_ > safe_dt)
		{
			dt_ = safe_dt;
		}

	}

	void updatedt(nTuple<Real, ndims> const & kimg)
	{
		updatedt(0.0);
	}
	void updatedt(nTuple<Complex, ndims> const & kimg)
	{
		Real dx2 = 0.0;

		if (std::imag(kimg[XAxis]) > EPSILON)
		{
			dx2 += TWOPI * TWOPI
					/ (std::imag(kimg[XAxis]) * std::imag(kimg[XAxis]));
		}
		if (std::imag(kimg[ZAxis]) > EPSILON)
		{
			dx2 += TWOPI * TWOPI
					/ (std::imag(kimg[ZAxis]) * std::imag(kimg[ZAxis]));
		}
		if (std::imag(kimg[YAxis]) > EPSILON)
		{
			dx2 += TWOPI * TWOPI
					/ (std::imag(kimg[YAxis]) * std::imag(kimg[YAxis]));
		}

		updatedt(dx2);

	}

	template<typename TDict, typename ...Others>
	bool load(TDict const & dict, Others &&...others)
	{

		if (topology_type::load(dict) && dict["Min"] && dict["Max"])
		{

			LOGGER << "Load CartesianGeometry ";

			extents(

			dict["Min"].template as<nTuple<Real, ndims>>(),

			dict["Max"].template as<nTuple<Real, ndims>>());

			CFL_ = dict["CFL"].template as<Real>(0.5);

			dt_ = dict["dt"].template as<Real>(1.0);

			return true;
		}

		WARNING << "Configure Error: no Min or Max ";

		return false;

	}

	std::string save(std::string const &path) const
	{
		return topology_type::save(path);

	}

	template<typename OS>
	OS & print(OS &os) const
	{
		topology_type::print(os);

		os

		<< " Min = " << xmin_ << " ," << std::endl

		<< " Max  = " << xmax_ << "," << std::endl

		<< " dt  = " << dt_ << "," << std::endl;

		return os;

	}

	void extents(nTuple<Real, ndims> pmin, nTuple<Real, ndims> pmax)
	{
		xmin_ = pmin;
		xmax_ = pmax;
	}

	inline auto extents() const
	DECL_RET_TYPE (std::make_pair(xmin_, xmax_))

	inline coordinates_type dx(index_type s = 0UL) const
	{
		coordinates_type res;

		auto d = topology_type::dx();

		for (size_t i = 0; i < ndims; ++i)
		{
			res[i] = length_[i] * d[i];
		}

		return std::move(res);
	}

	template<typename ... Args>
	coordinates_type coordinates(Args && ... args) const
	{
		return std::move(
				coordinates_from_topology(
						topology_type::get_coordinates(
								std::forward<Args >(args)...)));
	}

	coordinates_type coordinates_from_topology(coordinates_type const &x) const
	{

		return coordinates_type(
		{

		x[0] * length_[0] + shift_[0],

		x[1] * length_[1] + shift_[1],

		x[2] * length_[2] + shift_[2]

		});

	}
	coordinates_type coordinates_to_topology(coordinates_type const &x) const
	{
		return coordinates_type(
		{

		(x[0] - shift_[0]) * inv_length_[0],

		(x[1] - shift_[1]) * inv_length_[1],

		(x[2] - shift_[2]) * inv_length_[2]

		});

	}

	template<typename TI>
	inline auto index_to_coordinates(TI const&idx) const
	DECL_RET_TYPE((coordinates_from_topology(
							topology_type::index_to_coordinates(idx))))

	inline auto coordinates_to_index(coordinates_type const & x) const
	DECL_RET_TYPE((topology_type::coordinates_to_index(
							coordinates_to_topology(x))))

	/**
	 * @bug: truncation error of coordinates transform larger than 1000 epsilon (1e4 epsilon for cylindrical coordiantes)
	 * @param args
	 * @return
	 */
	template<typename ... Args>
	inline coordinates_type coordinates_local_to_global(Args && ... args) const
	{
		return std::move(
				coordinates_from_topology(
						topology_type::coordinates_local_to_global(
								std::forward<Args >(args)...)));
	}

	std::tuple<index_type, coordinates_type> coordinates_global_to_local(
			coordinates_type x, index_type shift = 0UL) const
	{
		return std::move(
				topology_type::coordinates_global_to_local(
						std::move(coordinates_to_topology(x)), shift));
	}
	std::tuple<index_type, coordinates_type> coordinates_global_to_local_NGP(
			coordinates_type x, index_type shift = 0UL) const
	{
		return std::move(
				topology_type::coordinates_global_to_local_NGP(
						std::move(coordinates_to_topology(x)), shift));
	}

	coordinates_type InvMapTo(coordinates_type const &y) const
	{
		coordinates_type x;

		x[CARTESIAN_XAXIS] = y[XAxis];
		x[CARTESIAN_YAXIS] = y[YAxis];
		x[CARTESIAN_ZAXIS] = y[ZAxis];

		return std::move(x);
	}

	coordinates_type MapTo(coordinates_type const &x) const
	{

		coordinates_type y;

		y[XAxis] = x[CARTESIAN_XAXIS];
		y[YAxis] = x[CARTESIAN_YAXIS];
		y[ZAxis] = x[CARTESIAN_ZAXIS];

		return std::move(y);
	}

	template<typename TV>
	std::tuple<coordinates_type, TV> push_forward(
			std::tuple<coordinates_type, TV> const & Z) const
	{
		return std::move(std::make_tuple(MapTo(std::get<0>(Z)), std::get<1>(Z)));
	}

	template<typename TV>
	std::tuple<coordinates_type, TV> pull_back(
			std::tuple<coordinates_type, TV> const & R) const
	{
		return std::move(
				std::make_tuple(InvMapTo(std::get<0>(R)), std::get<1>(R)));
	}

	/**
	 *
	 *   transform vector from Cartesian to Cartesian
	 *
	 * @param Z \f$\left(x,v\right)\f$\f$ u[XAixs] \partial_x +  u[YAixs] \partial_y + u[ZAixs] \partial_z \f$
	 * @param CartesianZAxis
	 * @return y u
	 *
	 */

	template<typename TV>
	std::tuple<coordinates_type, nTuple<TV, ndims> > push_forward(
			std::tuple<coordinates_type, nTuple<TV, ndims> > const & Z) const
	{
		coordinates_type y = MapTo(std::get<0>(Z));

		auto const & v = std::get<1>(Z);

		nTuple<TV, ndims> u;

		u[XAxis] = v[CARTESIAN_XAXIS];
		u[YAxis] = v[CARTESIAN_YAXIS];
		u[ZAxis] = v[CARTESIAN_ZAXIS];

		return (std::make_tuple(y, u));
	}

	/**
	 *
	 *  transform vector  from    Cylindrical to Cartesian
	 * @param R \f$\left(r, z ,\theta\right)\f$
	 *  \f$ u = u[RAixs] \partial_r +  u[1]  r[RAxis] \partial_theta + u[ZAixs] \partial_z\f$
	 * @param CartesianZAxis
	 * @return  x,\f$v = v[XAixs] \partial_x +  v[YAixs] \partial_y + v[ZAixs] \partial_z\f$
	 */
	template<typename TV>
	std::tuple<coordinates_type, nTuple<TV, ndims> > pull_back(
			std::tuple<coordinates_type, nTuple<TV, ndims> > const & R,
			size_t CartesianZAxis = 2) const
	{
		auto x = InvMapTo(std::get<0>(R));
		auto const & u = std::get<1>(R);

		nTuple<TV, ndims> v;

		v[CARTESIAN_XAXIS] = u[XAxis];
		v[CARTESIAN_YAXIS] = u[YAxis];
		v[CARTESIAN_ZAXIS] = u[ZAxis];

		return std::move(std::make_tuple(x, v));
	}

	template<typename TR>
	auto select(TR range, coordinates_type const & xmin,
			coordinates_type const & xmax) const
					DECL_RET_TYPE((topology_type::select(range, this->coordinates_to_topology(xmin),this->coordinates_to_topology(xmax))))

	template<typename TR, typename ...Args>
	auto select(TR range, Args && ...args) const
	DECL_RET_TYPE((topology_type::select(range,std::forward<Args >(args)...)))

	auto select(size_t iform) const
	DECL_RET_TYPE((this->topology_type::select(iform)))

	template<typename TV>
	TV Sample(std::integral_constant<size_t, VERTEX>, index_type s,
			TV const &v) const
	{
		return v;
	}

	template<typename TV>
	TV Sample(std::integral_constant<size_t, VOLUME>, index_type s,
			TV const &v) const
	{
		return v;
	}

	template<typename TV>
	TV Sample(std::integral_constant<size_t, EDGE>, index_type s,
			nTuple<TV, 3> const &v) const
	{
		return v[topology_type::component_number(s)];
	}

	template<typename TV>
	TV Sample(std::integral_constant<size_t, FACE>, index_type s,
			nTuple<TV, 3> const &v) const
	{
		return v[topology_type::component_number(s)];
	}

	template<size_t IFORM, typename TV>
	TV Sample(std::integral_constant<size_t, IFORM>, index_type s,
			TV const & v) const
	{
		return v;
	}

//***************************************************************************************************
// Volume
//***************************************************************************************************

	/**
	 *\verbatim
	 *                ^y
	 *               /
	 *        z     /
	 *        ^    /
	 *        |  110-------------111
	 *        |  /|              /|
	 *        | / |             / |
	 *        |/  |            /  |
	 *       100--|----------101  |
	 *        | m |           |   |
	 *        |  010----------|--011
	 *        |  /            |  /
	 *        | /             | /
	 *        |/              |/
	 *       000-------------001---> x
	 *
	 *\endverbatim
	 */

	scalar_type volume_[8] =
	{ 1, // 000
			1, //001
			1, //010
			1, //011
			1, //100
			1, //101
			1, //110
			1  //111
			};
	scalar_type inv_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

	scalar_type dual_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

	scalar_type inv_dual_volume_[8] =
	{ 1, 1, 1, 1, 1, 1, 1, 1 };

public:

	scalar_type cell_volume(index_type s) const
	{
		return topology_type::cell_volume(s) * volume_[1] * volume_[2]
				* volume_[4];
	}
	scalar_type volume(index_type s) const
	{
		return topology_type::volume(s) * volume_[topology_type::node_id(s)];
	}
	scalar_type inv_volume(index_type s) const
	{
		return topology_type::inv_volume(s)
				* inv_volume_[topology_type::node_id(s)];
	}

	scalar_type dual_volume(index_type s) const
	{
		return topology_type::dual_volume(s)
				* dual_volume_[topology_type::node_id(s)];
	}
	scalar_type inv_dual_volume(index_type s) const
	{
		return topology_type::inv_dual_volume(s)
				* inv_dual_volume_[topology_type::node_id(s)];
	}

	Real HodgeStarVolumeScale(index_type s) const
	{
		return 1.0;
	}

}
;
template<typename TTopology, size_t ZAXIS>
bool CartesianCoordinates<TTopology, ZAXIS>::update()
{

	topology_type::update();

	if (!topology_type::is_ready())
	{
		ERROR("topology initialize failed!");
	}

	auto dims = topology_type::dimensions();

	for (size_t i = 0; i < ndims; ++i)
	{
		shift_[i] = xmin_[i];

		if ((xmax_[i] - xmin_[i]) < EPSILON || dims[i] <= 1)
		{

			xmax_[i] = xmin_[i];

			inv_length_[i] = 0.0;

			length_[i] = 0.0;

			volume_[1UL << (ndims - i - 1)] = 1.0;

			dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			inv_volume_[1UL << (ndims - i - 1)] = 1.0;

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

		}
		else
		{
			inv_length_[i] = 1.0 / (xmax_[i] - xmin_[i]);

			length_[i] = (xmax_[i] - xmin_[i]);

			volume_[1UL << (ndims - i - 1)] = length_[i];

			dual_volume_[7 - (1UL << (ndims - i - 1))] = length_[i];

			inv_volume_[1UL << (ndims - i - 1)] = inv_length_[i];

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = inv_length_[i];

		}
	}

	/**
	 *\verbatim
	 *                ^y
	 *               /
	 *        z     /
	 *        ^    /
	 *        |  110-------------111
	 *        |  /|              /|
	 *        | / |             / |
	 *        |/  |            /  |
	 *       100--|----------101  |
	 *        | m |           |   |
	 *        |  010----------|--011
	 *        |  /            |  /
	 *        | /             | /
	 *        |/              |/
	 *       000-------------001---> x
	 *
	 *\endverbatim
	 */

	volume_[0] = 1;
//		volume_[1] /* 001 */= dx_[0];
//		volume_[2] /* 010 */= dx_[1];
//		volume_[4] /* 100 */= dx_[2];

	volume_[3] /* 011 */= volume_[1] * volume_[2];
	volume_[5] /* 101 */= volume_[4] * volume_[1];
	volume_[6] /* 110 */= volume_[2] * volume_[4];

	volume_[7] /* 111 */= volume_[1] * volume_[2] * volume_[4];

	dual_volume_[7] = 1;
//		dual_volume_[6] /* 001 */= dx_[0];
//		dual_volume_[5] /* 010 */= dx_[1];
//		dual_volume_[3] /* 100 */= dx_[2];

	dual_volume_[4] /* 011 */= dual_volume_[6] * dual_volume_[5];
	dual_volume_[2] /* 101 */= dual_volume_[3] * dual_volume_[6];
	dual_volume_[1] /* 110 */= dual_volume_[5] * dual_volume_[3];

	dual_volume_[0] /* 111 */= dual_volume_[6] * dual_volume_[5]
			* dual_volume_[3];

	inv_volume_[0] = 1;
//		inv_volume_[1] /* 001 */= inv_dx_[0];
//		inv_volume_[2] /* 010 */= inv_dx_[1];
//		inv_volume_[4] /* 100 */= inv_dx_[2];

	inv_volume_[3] /* 011 */= inv_volume_[1] * inv_volume_[2];
	inv_volume_[5] /* 101 */= inv_volume_[4] * inv_volume_[1];
	inv_volume_[6] /* 110 */= inv_volume_[2] * inv_volume_[4];

	inv_volume_[7] /* 111 */= inv_volume_[1] * inv_volume_[2] * inv_volume_[4];

	inv_dual_volume_[7] = 1;
//		inv_dual_volume_[6] /* 001 */= inv_dx_[0];
//		inv_dual_volume_[5] /* 010 */= inv_dx_[1];
//		inv_dual_volume_[3] /* 100 */= inv_dx_[2];

	inv_dual_volume_[4] /* 011 */= inv_dual_volume_[6] * inv_dual_volume_[5];
	inv_dual_volume_[2] /* 101 */= inv_dual_volume_[3] * inv_dual_volume_[6];
	inv_dual_volume_[1] /* 110 */= inv_dual_volume_[5] * inv_dual_volume_[3];

	inv_dual_volume_[0] /* 111 */= inv_dual_volume_[6] * inv_dual_volume_[5]
			* inv_dual_volume_[3];

	updatedt();

	is_ready_ = true;

	return is_ready_;
}

}  // namespace simpla

#endif /* COORDINATES_CARTESIAN_H_ */

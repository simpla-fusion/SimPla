/**
 * @file   cartesian.h
 *
 *  created on: 2014-3-13
 *      Author: salmon
 */

#ifndef CORE_MESH_STRUCTURED_COORDINATES_CARTESIAN_H_
#define CORE_MESH_STRUCTURED_COORDINATES_CARTESIAN_H_

#include <stddef.h>
#include <string>
#include <tuple>
#include <type_traits>
#include "../../../utilities/utilities.h"
#include "../../../physics/constants.h"
#include "../../../physics/physical_constants.h"
#include "../../../gtl/enable_create_from_this.h"
#include "../../mesh.h"

namespace simpla
{
/**
 *  @ingroup geometry
 *
 *  \brief  Cartesian coordinates (X Y Z)
 *
 */
template<typename TTopology, size_t ZAXIS = CARTESIAN_ZAXIS>
struct CartesianCoordinates: public TTopology, public enable_create_from_this<
										CartesianCoordinates<TTopology, ZAXIS>>
{

public:
	typedef TTopology topology_type;
	typedef CartesianCoordinates<topology_type> this_type;
	typedef std::shared_ptr<this_type> holder_type;
	static constexpr size_t ndims = topology_type::ndims;

	static constexpr size_t XAxis = (ZAXIS + 1) % 3;
	static constexpr size_t YAxis = (ZAXIS + 2) % 3;
	static constexpr size_t ZAxis = ZAXIS;

	typedef typename topology_type::coordinates_type coordinates_type;
	typedef typename topology_type::id_type id_type;
	typedef Real scalar_type;

private:
	bool is_valid_ = false;

	coordinates_type m_xmin_ /*= { 0, 0, 0 }*/;

	coordinates_type m_xmax_ /*= { 1, 1, 1 }*/;

	coordinates_type m_inv_length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type m_length_ /*= { 1.0, 1.0, 1.0 }*/;

	coordinates_type m_shift_/* = { 0, 0, 0 }*/;

	coordinates_type m_to_topology_factor_;

	coordinates_type m_from_topology_factor_;

public:
	CartesianCoordinates()
			: topology_type(), is_valid_(false),

			m_xmin_( { 0, 0, 0 }),

			m_xmax_( { 1, 1, 1 }),

			m_inv_length_( { 1.0, 1.0, 1.0 }),

			m_length_( { 1.0, 1.0, 1.0 }),

			m_shift_( { 0, 0, 0 })
	{

	}

	CartesianCoordinates(CartesianCoordinates const & other)
			: topology_type(other), m_xmin_(other.m_xmin_), m_xmax_(
					other.m_xmax_) //, m_dx_(other.m_dx_),

	{

	}
	~CartesianCoordinates()
	{
	}

	void swap(CartesianCoordinates & other)
	{
		topology_type::swap(other);
	}

	static std::string get_type_as_string_static()
	{
		return "Cartesian";
	}

	std::string get_type_as_string() const
	{
		return get_type_as_string_static();
	}

	this_type & self()
	{
		return *this;
	}
	this_type const& self() const
	{
		return *this;
	}

	struct Hash
	{
		holder_type self_;

		Hash(holder_type g)
				: self_(g)
		{
		}
		Hash(Hash const & other)
				: self_(other.self_)
		{
		}
		~Hash()
		{
		}

	private:
		template<typename T>
		constexpr size_t hash_(T const &x,
				std::integral_constant<bool, false>) const
		{
			return self_->coordiantes_to_id(x);
		}
		template<typename T>
		constexpr size_t hash_(T const &p,
				std::integral_constant<bool, true>) const
		{
			return self_->coordiantes_to_id(p.x);
		}

		HAS_MEMBER(x);
	public:
		template<typename T>
		constexpr size_t operator()(T const &x) const
		{
			return hash_(x,std::integral_constant<bool, has_member_x<T>::value>());
		}

	};

	Hash hash_function() const
	{
		return Hash(enable_create_from_this<
				CartesianCoordinates<TTopology, ZAXIS>>::shared_from_this());
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
	void time(Real p_time)
	{
		time0_ = p_time;
	}
	Real time() const
	{
		return topology_type::time() * dt_ + time0_;
	}

	void dt(Real p_dt)
	{
		dt_ = p_dt;
	}

	Real dt() const
	{
		return dt_;
	}

	bool is_valid() const
	{
		return is_valid_ && topology_type::is_valid();
	}

	void deploy();
	void sync()
	{
	}

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
		topology_type::load(dict,std::forward<Others>(others)...);

		if (!topology_type::is_valid())
		{
			RUNTIME_ERROR("Topology is not initialized!");
			return false;
		}

		if (!dict)
		{
			return false;
		}

		if (dict["Min"] && dict["Max"])
		{

			VERBOSE << "Load geometry : Cartesian " << std::endl;

			extents(

					dict["Min"].template as<nTuple<Real, ndims>>(),

					dict["Max"].template as<nTuple<Real, ndims>>());

			CFL_ = dict["CFL"].template as<Real>(0.5);

			dt_ = dict["dt"].template as<Real>(1.0);

		}
		else
		{
			WARNING << "Configure Error: no Min or Max ";

		}

		return true;
	}

	template<typename OS>
	OS & print(OS &os) const
	{
		topology_type::print(os);

		os << std::endl

		<< " Min = " << m_xmin_ << " ,"

		<< " Max  = " << m_xmax_ << "," << " dt  = " << dt_ << ",";

		return os;
	}

	template<typename T0,typename T1>
	void extents(T0 const& pmin, T1 const& pmax)
	{
		m_xmin_ = pmin;
		m_xmax_ = pmax;
	}

	inline auto extents() const
	DECL_RET_TYPE (std::make_pair(m_xmin_, m_xmax_))

	inline std::pair<coordinates_type,coordinates_type> local_extents() const
	{
		coordinates_type min,max,p_dx;

		std::tie(min,max)=topology_type::local_extents();
		p_dx=dx();
		min*=p_dx;min+=m_xmin_;
		max*=p_dx;max+=m_xmin_;

		return std::make_pair(min,max);

	}

	inline coordinates_type dx(id_type s = 0UL) const
	{
		coordinates_type res;

		auto d = topology_type::dimensions();

		for (size_t i = 0; i < ndims; ++i)
		{
			res[i] = m_length_[i] / d[i];
		}

		return std::move(res);
	}

	coordinates_type coordinates(id_type const & s) const
	{
		return std::move(
				coordinates_from_topology(topology_type::id_to_coordinates(s)));
	}

	coordinates_type coordinates_from_topology(coordinates_type const &x) const
	{

		return coordinates_type(
				{

					x[0] * m_from_topology_factor_[0] + m_shift_[0],

					x[1] * m_from_topology_factor_[1] + m_shift_[1],

					x[2] * m_from_topology_factor_[2] + m_shift_[2]

				});

	}
	coordinates_type coordinates_to_topology(coordinates_type const &x) const
	{
		return coordinates_type(
				{

					(x[0] - m_shift_[0]) * m_to_topology_factor_[0],

					(x[1] - m_shift_[1]) * m_to_topology_factor_[1],

					(x[2] - m_shift_[2]) * m_to_topology_factor_[2]

				});

	}

	template<typename TI>
	inline auto index_to_coordinates(TI const&idx) const
	DECL_RET_TYPE((coordinates_from_topology(
							topology_type::index_to_coordinates(idx))))

	inline auto coordinates_to_index(coordinates_type const & x) const
	DECL_RET_TYPE((topology_type::coordinates_to_index(
							coordinates_to_topology(x))))

	inline id_type coordinates_to_id(coordinates_type const &x)const
	{
		return topology_type::coordinates_to_id(coordinates_to_topology(x));
	}
	/**
	 * @bug: truncation error of coordinates transform larger than 1000
	 *     epsilon (1e4 epsilon for cylindrical coordinates)
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

	std::tuple<id_type, coordinates_type> coordinates_global_to_local(
			coordinates_type x, id_type shift = 0UL) const
	{
		return std::move(
				topology_type::coordinates_global_to_local(
						std::move(coordinates_to_topology(x)), shift));
	}
	std::tuple<id_type, coordinates_type> coordinates_global_to_local_NGP(
			coordinates_type x, id_type shift = 0UL) const
	{
		return std::move(
				topology_type::coordinates_global_to_local_NGP(
						std::move(coordinates_to_topology(x)), shift));
	}

	nTuple<Real, 3> MapToCartesian(coordinates_type const &y) const
	{
		nTuple<Real, 3> x;

		x[CARTESIAN_XAXIS] = y[XAxis];
		x[CARTESIAN_YAXIS] = y[YAxis];
		x[CARTESIAN_ZAXIS] = y[ZAxis];

		return std::move(x);
	}

	coordinates_type MapFromCartesian(nTuple<Real, 3> const &x) const
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
		return std::move(
				std::make_tuple(MapFromCartesian(std::get<0>(Z)),
						std::get<1>(Z)));
	}

	template<typename TV>
	std::tuple<coordinates_type, TV> pull_back(
			std::tuple<coordinates_type, TV> const & R) const
	{
		return std::move(
				std::make_tuple(MapToCartesian(std::get<0>(R)), std::get<1>(R)));
	}

	template<typename TD>
	auto gather(TD const & d, coordinates_type const & x) const
	DECL_RET_TYPE(topology_type::gather(d,coordinates_to_topology(x)))

	template<typename TD, typename TV>
	void scatter(TD & d, coordinates_type const &x, TV const & v) const
	{
		(topology_type::scatter(d, coordinates_to_topology(x), v));
	}

	coordinates_type Lie_trans(coordinates_type const & x,
			nTuple<Real, 3> const & v)
	{
		coordinates_type res;
		res = x + v;

		return std::move(res);
	}

	coordinates_type Lie_trans(
			std::tuple<coordinates_type, nTuple<Real, 3> > const &Z,
			nTuple<Real, 3> const & v)
	{
		coordinates_type res;
		res = std::get<0>(Z) + v;

		return std::move(res);
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
		coordinates_type y = MapFromCartesian(std::get<0>(Z));

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
		auto x = MapToCartesian(std::get<0>(R));
		auto const & u = std::get<1>(R);

		nTuple<TV, ndims> v;

		v[CARTESIAN_XAXIS] = u[XAxis];
		v[CARTESIAN_YAXIS] = u[YAxis];
		v[CARTESIAN_ZAXIS] = u[ZAxis];

		return std::move(std::make_tuple(x, v));
	}

//	template<size_t IFORM, typename TR>
//	auto select(TR range, coordinates_type const & xmin,
//			coordinates_type const & xmax) const
//			DECL_RET_TYPE((topology_type::template select<IFORM>(range,
//									this->coordinates_to_topology(xmin),
//									this->coordinates_to_topology(xmax))))
//
//	template<size_t IFORM, typename TR, typename ...Args>
//	auto select(TR range, Args && ...args) const
//	DECL_RET_TYPE((topology_type::template select<IFORM>(
//							range,std::forward<Args >(args)...)))
//
//	template<size_t IFORM>
//	auto select() const
//	DECL_RET_TYPE((this->topology_type:: template select<IFORM>()))

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
	{	1, // 000
		1,//001
		1,//010
		1,//011
		1,//100
		1,//101
		1,//110
		1//111
	};
	scalar_type inv_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

	scalar_type dual_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

	scalar_type inv_dual_volume_[8] =
	{	1, 1, 1, 1, 1, 1, 1, 1};

public:

	constexpr scalar_type cell_volume(id_type s) const
	{
		return volume_[1] * volume_[2] * volume_[4];
	}
	constexpr scalar_type volume(id_type s) const
	{
		return volume_[topology_type::node_id(s)];
	}
	constexpr scalar_type inv_volume(id_type s) const
	{
		return inv_volume_[topology_type::node_id(s)];
	}

	constexpr scalar_type dual_volume(id_type s) const
	{
		return dual_volume_[topology_type::node_id(s)];
	}
	constexpr scalar_type inv_dual_volume(id_type s) const
	{
		return inv_dual_volume_[topology_type::node_id(s)];
	}

	constexpr Real HodgeStarVolumeScale(id_type s) const
	{
		return 1.0;
	}

}
;
template<typename TTopology, size_t ZAXIS>
void CartesianCoordinates<TTopology, ZAXIS>::deploy()
{

	topology_type::deploy();

	if (!topology_type::is_valid())
	{
		ERROR("topology initialize failed!");
	}

	auto dims = topology_type::dimensions();

	for (size_t i = 0; i < ndims; ++i)
	{

		if ((m_xmax_[i] - m_xmin_[i]) < EPSILON)
			dims[i] = 1;
	}

	topology_type::dimensions(&dims[0]);

	topology_type::deploy();

	for (size_t i = 0; i < ndims; ++i)
	{
		m_shift_[i] = m_xmin_[i];

		if (dims[i] <= 1)
		{

			m_xmax_[i] = m_xmin_[i];

			m_inv_length_[i] = 0.0;

			m_length_[i] = 0.0;

			volume_[1UL << (ndims - i - 1)] = 1.0;

			dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			inv_volume_[1UL << (ndims - i - 1)] = 1.0;

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = 1.0;

			m_to_topology_factor_[i] = 0;
			m_from_topology_factor_[i] = 0;
		}
		else
		{
			m_inv_length_[i] = 1.0 / (m_xmax_[i] - m_xmin_[i]);

			m_length_[i] = (m_xmax_[i] - m_xmin_[i]);

			volume_[1UL << (ndims - i - 1)] = m_length_[i];

			dual_volume_[7 - (1UL << (ndims - i - 1))] = m_length_[i];

			inv_volume_[1UL << (ndims - i - 1)] = m_inv_length_[i];

			inv_dual_volume_[7 - (1UL << (ndims - i - 1))] = m_inv_length_[i];

			m_to_topology_factor_ = dims[i] / (m_xmax_[i] - m_xmin_[i]);

			m_from_topology_factor_[i] = (m_xmax_[i] - m_xmin_[i]) / dims[i];

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

	is_valid_ = true;

}

}  // namespace simpla

#endif /* CORE_MESH_STRUCTURED_COORDINATES_CARTESIAN_H_ */

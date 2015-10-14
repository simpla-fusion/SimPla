/**
 * @file constant_metric.h.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_CONSTANT_METRIC_H_H
#define SIMPLA_CONSTANT_METRIC_H_H

#include "../../gtl/primitives.h"
#include "../../gtl/macro.h"
#include "../../gtl/type_traits.h"

#include "../topology/topology.h"

#include "coordinate_system.h"

namespace simpla
{

namespace geometry
{

template<typename ...> struct Geometry;

template<typename CS, typename TopologyTags>
struct Geometry<CS, topology::Topology<TopologyTags> > : public topology::Topology<TopologyTags>
{
	typedef CS cs_type;

	mertic<cs_type> m_metric_;

	typedef topology::Topology<TopologyTags> topology_type;

	static constexpr int ndims = topology_type::ndims;

	typedef nTuple<Real, ndims> topology_point_type;


	typedef traits::scalar_type_t<cs_type> scalar_type;

	typedef traits::point_type_t<cs_type> point_type;

	typedef traits::vector_type_t<cs_type> vector_type;

private:

	topology_point_type m_from_topology_orig_ = {0, 0, 0};

	topology_point_type m_to_topology_orig_ = {0, 0, 0};

	point_type m_coords_min_ = {0, 0, 0};

	point_type m_coords_max_ = {1, 1, 1};

	vector_type m_dx_; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

	vector_type m_delta_; //!< equiv. m_dx_, except m_delta_[i]=1 when m_dims_[i]==1

	topology_point_type m_to_topology_scale_;

	topology_point_type m_from_topology_scale_;
public:


	std::string get_type_as_string() const
	{
		return "Geometry Unnamed";
	}


	point_type epsilon() const
	{
		return topology_type::EPSILON * m_from_topology_scale_;
	}

	template<typename ...Args>
	void extents(Args &&...args)
	{
		m_coords_min_ = traits::unpack_args<0>(std::forward<Args>(args)...);
		m_coords_max_ = traits::unpack_args<1>(std::forward<Args>(args)...);
	}

	template<typename T0>
	void extents(T0 const &box)
	{
		extents(traits::get<0>(box), traits::get<1>(box));
	}

	constexpr std::pair<point_type, point_type> extents() const
	{
		return (std::make_pair(m_coords_min_, m_coords_max_));
	}

//	constexpr std::pair<point_type, point_type> local_extents() const
//	{
//		return (std::make_pair(point(m_id_local_min_), point(m_id_local_max_)));
//	}

	vector_type const &dx() const
	{
		return m_dx_;
	}

/**
 * @name  Coordinate map
 * @{
 *
 *        Topology Manifold       Geometry Manifold
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
	point_type point(id_type const &s) const { return std::move(map(topology_type::coordinates(s))); }

	point_type map(topology_point_type const &x) const
	{

		point_type res;


		res[0] = std::fma(traits::get<0>(x), m_from_topology_scale_[0],
				m_from_topology_orig_[0]);

		res[1] = std::fma(traits::get<1>(x), m_from_topology_scale_[1],
				m_from_topology_orig_[1]);

		res[2] = std::fma(traits::get<2>(x), m_from_topology_scale_[2],
				m_from_topology_orig_[2]);


		return std::move(res);
	}

	topology_point_type inv_map(point_type const &y) const
	{

		topology_point_type res;


		res[0] = std::fma(traits::get<0>(y), m_to_topology_scale_[0],
				m_to_topology_orig_[0]);

		res[1] = std::fma(traits::get<1>(y), m_to_topology_scale_[1],
				m_to_topology_orig_[1]);

		res[2] = std::fma(traits::get<2>(y), m_to_topology_scale_[2],
				m_to_topology_orig_[2]);

		return std::move(res);
	}


/** @} */
/** @name Volume
 * @{
 */

	Real m_volume_[9];
	Real m_inv_volume_[9];
	Real m_dual_volume_[9];
	Real m_inv_dual_volume_[9];

	Real volume_(id_type s) const
	{
		return m_metric_.dual_volume(topology_type::node_id(s), point(s),
				m_delta_);
	}

	Real dual_volume_(id_type s) const
	{

		return m_metric_.dual_volume(topology_type::node_id(s), point(s),
				m_delta_);
	}

	Real volume(id_type s) const
	{
		return m_volume_[topology_type::node_id(s)] * volume_(s);
	}

	Real dual_volume(id_type s) const
	{
		return m_dual_volume_[topology_type::node_id(s)] * dual_volume_(s);
	}

	Real inv_volume(id_type s) const
	{
		return m_inv_volume_[topology_type::node_id(s)] / volume_(s);
	}

	Real inv_dual_volume(id_type s) const
	{
		return m_inv_dual_volume_[topology_type::node_id(s)] / dual_volume_(s);
	}

/**@}*/



	template<typename TFun>
	auto pull_back(point_type const &x, TFun const &fun) const

	DECL_RET_TYPE((fun(map(x))))

	template<typename TFun>
	auto push_forward(point_type const &y, TFun const &fun) const

	DECL_RET_TYPE((fun(inv_map(y))))

	Vec3 pull_back(point_type const &y, vector_type const &u) const { return inv_map(y + u) - inv_map(y); }

	Vec3 push_forward(point_type const &x, vector_type const &v) const { return map(x + v) - map(x); }

	template<typename TX>
	topology_point_type coordinates_to_topology(TX const &y) const { return inv_map(y); }

	template<typename TX>
	point_type coordinates_from_topology(TX const &x) const { return map(x); }

/**
 * @bug: truncation error of coordinates transform larger than 1000
 *     epsilon (1e4 epsilon for cylindrical coordinates)
 * @param args
 * @return
 */
	point_type coordinates_local_to_global(std::tuple<id_type, topology_point_type> const &t) const
	{
		return std::move(map(topology_type::coordinates_local_to_global(t)));
	}

	std::tuple<id_type, topology_point_type> coordinates_global_to_local(
			point_type x, int n_id = 0) const
	{
		return std::move(topology_type::coordinates_global_to_local(coordinates_to_topology(x), n_id));
	}
}; //struct Geometry<CS, topology::Topology<TopologyTags> >

}//namespace geometry
}//namespace simpla
#endif //SIMPLA_CONSTANT_METRIC_H_H

/*
 * @file surface.h
 *
 *  created on: 2014-4-21
 *      Author: salmon
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <map>
#include <tuple>

#include "../utilities/utilities.h"
#include "../utilities/type_traits.h"
#include "../numeric/geometric_algorithm.h"
namespace simpla
{

/**
 *  @ingroup Model
 */

template<typename TM>
class Surface: public std::map<typename TM::id_type, std::tuple<Real, Vec3>>
{
public:
	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::range_type range_type;

	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	typedef Surface<mesh_type> this_type;

	typedef std::map<id_type, std::tuple<Real, Vec3>> base_type;

	mesh_type const & m_mesh_;
	range_type m_range_;
	id_type m_nid_;

	Surface(mesh_type const & m)
			: m_mesh_(m), m_radius_(std::sqrt(inner_product(m.dx(), m.dx())))
	{
	}

	~Surface()
	{
	}

	auto box() const
	DECL_RET_TYPE((m_range_.box()))

	template<typename ...Args>
	auto box(Args &&...args) const
	{
		m_range_.reset(std::forward<Args>(args));
	}

	std::tuple<Real, Vec3> distance(coordinate_tuple const & x) const
	{
		Real dist;
		Vec3 normal;

		return std::make_tuple(dist, normal);
	}

	void fix_to_bound()
	{
		m_mesh_.find_bound(*this,
				[&](typename base_type::value_type const& item)
				{
					return item.first;
				}).swap(m_range_);
	}

	template<typename TFun>
	void cut(TFun const & surface);

	bool divide(id_type s)
	{
		UNIMPLEMENTED;
		return false;
	}
	bool merge(id_type s)
	{
		UNIMPLEMENTED;
		return false;
	}

	template<typename TDict>
	void load_surface(TDict const & dict);
};
template<typename TM>
template<typename TDict>
void Surface<TM>::load_surface(TDict const & dict)
{

	if (dict["Box"])
	{
		std::vector<coordinate_tuple> points;

		dict["Box"].as(&points);

		m_range_.reset(m_mesh_.coordinates_to_topology(points[0]),
				m_mesh_.coordinates_to_topology(points[1]));

	}

	if (dict["Polygon"].is_function())
	{
		std::vector<coordinate_tuple> points

		dict["Polygon"].as(&points);
		for (id_type s : m_range_)
		{

		}

	}
}

}
// namespace simpla

#endif /* SURFACE_H_ */

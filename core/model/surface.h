/*
 * surface.h
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

/**
 *  \brief surface
 *
 */
template<typename TM>
class Surface: public std::map<typename TM::id_type, std::tuple<Real, Vec3>>
{
public:
	typedef TM mesh_type;

	typedef typename mesh_type::id_type id_type;

	typedef typename mesh_type::range_type range_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef Surface<mesh_type> this_type;

	typedef std::map<id_type, std::tuple<Real, Vec3>> base_type;

	mesh_type const & m_mesh_;
	range_type m_range_;
	id_type m_nid_;
	Surface(mesh_type const & m) :
			m_mesh_(m), m_radius_(std::sqrt(inner_product(m.dx(), m.dx())))
	{
	}
	~Surface()
	{
	}

	auto box() const
	DECL_RET_TYPE((m_range_.box()))

	template<typename TRange, typename TFun>
	void add(TRange const & r, TFun const & distance)
	{
		for (auto s : r)
		{
			auto x0 = m_mesh_.coordinates(s);

			auto res = distance(x0);

			id_type s1 = std::get<0>(
					m_mesh_.coordinates_global_to_local(
							x0 + std::get<0>(*it) * std::get<1>(*it), m_nid_));
			//if intersection point in the cell s
			if (s1 == s)
			{
				this->emplace(std::forward_as_tuple(s, std::move(res)));
			}

		}

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
};

}
// namespace simpla

#endif /* SURFACE_H_ */

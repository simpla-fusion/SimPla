/**
 * @file polygon.h
 *
 * @date 2015-5-12
 * @author salmon
 */

#ifndef CORE_GEOMETRY_POLYGON_H_
#define CORE_GEOMETRY_POLYGON_H_
#include "../gtl/iterator/sp_iterator_cycle.h"
namespace simpla
{

template<typename TM>
struct PolyGon: public std::list<coordinate_tuple>
{
	typedef TM mesh_type;
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;
	typedef typename mesh_type::id_type id_type;
	typedef typename mesh_type::topology_type topology_type;

	typedef std::list<coordinate_tuple> container_type;

	using typename container_type::iterator;

//	typedef CycleIterator<base_iterator> iterator;

	template<typename TDict>
	void load(TDict const & dict)
	{
		dict["Points"].as(this);
		dict["ZAxis"].as(&ZAXIS);

		for (auto & v : *this)
		{
			v = m_mesh_.coordinates_to_topology(v);
		}

	}

	/// @name PIP Point in Polygon,check a point in 2D polygon
	/// @{
private:
	mesh_type m_mesh_;
	//for point in polygon
	size_t ZAXIS = 2;
	std::list<Real> m_pip_constant_;
	std::list<Real> m_pip_multiple_;
public:

	void deploy_pip();
	template<typename T> bool check_inside(T const & coord) const;
	///@}

	/// @name distance
	/// @{
	/**
	 *
	 * @param x
	 * @return distance from point to polygon
	 */
	template<typename TX>
	std::tuple<Real, iterator> nearest_point(TX const & x) const;

	/**
	 * cut polygon by  a box (x0,x1)
	 * @param x
	 * @return distance from line segment to polygon
	 */
	bool cut(id_type s, std::list<coordinate_tuple>*);

	template<typename T0, typename T1>
	bool in_box(T0 const & x_min, T1 const & x_max);
	///@}

};

template<typename TM>
void PolyGon<TM>::deploy_pip()
{
	auto ib = container_type::begin();
	auto ie = container_type::end();

	auto vj = container_type::back();
	for (auto vi = container_type::begin(); vi != container_type::end(); ++vi)
	{
		if (vj[(ZAXIS + 2) % 3] == (*vi)[(ZAXIS + 2) % 3])
		{
			m_pip_constant_.push_back((*vi)[(ZAXIS + 1) % 3]);
			m_pip_multiple_.push_back(0);
		}
		else
		{
			m_pip_constant_.push_back(
					(*vi)[(ZAXIS + 1) % 3]
							- ((*vi)[(ZAXIS + 2) % 3] * vj[(ZAXIS + 1) % 3])
									/ (vj[(ZAXIS + 2) % 3]
											- (*vi)[(ZAXIS + 2) % 3])
							+ ((*vi)[(ZAXIS + 2) % 3] * (*vi)[(ZAXIS + 1) % 3])
									/ (vj[(ZAXIS + 2) % 3]
											- (*vi)[(ZAXIS + 2) % 3]));
			m_pip_multiple_.push_back(
					(vj[(ZAXIS + 1) % 3] - (*vi)[(ZAXIS + 1) % 3])
							/ (vj[(ZAXIS + 2) % 3] - (*vi)[(ZAXIS + 2) % 3]));
		}
		vj = *vi;
	}
}

template<typename TM>
template<typename T>
bool PolyGon<TM>::check_inside(T const & coord) const
{
	coordinate_tuple x0;
	x0 = m_mesh_.coordinates_to_topology(coord);

	auto const &y = x0[(ZAXIS + 2) % 3];
	auto const &x = x0[(ZAXIS + 1) % 3];

	bool oddNodes = false;

	auto vj = container_type::back();
	auto multiple_it = m_pip_multiple_.begin();
	auto constant_it = m_pip_constant_.begin();

	for (auto vi = container_type::begin(); vi != container_type::end();
			++vi, ++constant_it, ++multiple_it)

	{
		if ((((*vi)[(ZAXIS + 2) % 3] < y) && (vj[(ZAXIS + 2) % 3] >= y))
				|| ((vj[(ZAXIS + 2) % 3] < y) && ((*vi)[(ZAXIS + 2) % 3] >= y)))
		{
			oddNodes ^= (y * (*multiple_it) + (*constant_it) < x);
		}

		vj = *vi;
	}

	return oddNodes;
}

template<typename TM>
template<typename TX>
std::tuple<Real, typename PolyGon<TM>::iterator> PolyGon<TM>::nearest_point(
		TX const & coord) const
{
	coordinate_tuple x;
	x = m_mesh_.coordinates_to_topology(coord);

	Real min_dist2 = std::numeric_limits<Real>::max();

	Real res_s = 0;

	auto it = make_cycle_iterator(container_type::begin(),
			container_type::end());

	auto ie = container_type::end();

	Real dist;

	iterator res_it;

	coordinate_tuple p0, p1;

	p1 = *it;

	for (; it != ie; ++it)
	{
		p0 = p1;

		p1 = *(it++);

		Real s = nearest_point_to_line_segment(x, p0, p1);

		Real dist2 = inner_product(x - (p0 * (1 - s) + p1),
				x - (p0 * (1 - s) + p1));

		if (min_dist2 > dist2 || (min_dist2 == dist2 && s == 0))
		{
			res_it = it;
			res_s = s;
			min_dist2 = dist2;
		}
	}

	return std::make_tuple(res_s, res_it);

}
template<typename TM>
bool PolyGon<TM>::cut(id_type s, std::list<coordinate_tuple>* res)
{
	coordinate_tuple x0, x_min, x_max;
	x0 = topology_type::coordinates(s);
	x_min = topology_type::coordinates(s - topology_type::_DA);
	x_max = topology_type::coordinates(s + topology_type::_DA);

	auto it = make_cycle_iterator(container_type::begin(),
			container_type::end());

	auto ie = container_type::end();

	Real dist;

	iterator res_it;

	coordinate_tuple p0, p1;

	p1 = *it;

	for (; it != ie; ++it)
	{
		p0 = p1;

		p1 = *(it++);

		Real s = nearest_point_to_line_segment(x0, p0, p1);

		coordinate_tuple p = (p0 * (1 - s) + p1);

		Real dist = inner_product(x0 - p, x0 - p);

		if (dist > topology_type::COORDINATES_MESH_FACTOR)
		{
			continue;
		};

	}
}

}  // namespace simpla

#endif /* CORE_GEOMETRY_POLYGON_H_ */

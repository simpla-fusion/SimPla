/**
 * @file geometry_object.h
 *
 *  Created on: 2015年6月2日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEOMETRY_OBJECT_H_
#define CORE_GEOMETRY_GEOMETRY_OBJECT_H_

namespace simpla
{
namespace geometry_object
{
struct Sphere
{
	Real x, y, z;
	Real r;
};

struct Cylinder
{
	Real x0, y0, z0;
	Real x1, y1, z1;
	Real r;
};
struct Ring
{
	Real x0, y0, z0;
	Real x1, y1, z1;
	Real r;
};
struct Cylinder;
struct Tetrahedron;
struct Pyramid;
struct Polygons;

template<typename ...>struct ImplicitFunction;

template<>
struct ImplicitFunction<Sphere>
{
	Sphere m_object_;

	ImplicitFunction(nTuple<Real, 3> const & x0, Real radius)
			: m_object_( { x0, radius })
	{

	}
	~ImplicitFunction()
	{

	}
	Real operator()(nTuple<Real, 3> const & x)
	{
		return std::sqrt(inner_product(x - m_object_.x0, x - m_object_.x0))
				- m_object_.r;
	}
}

template<typename ...Args>
ImplicitFunction<Sphere> sphere(Args && ...args)
{
	return ImplicitFunction<Sphere>(std::forward<Args> (args)...);
}

}
// namespace simpla

#endif /* CORE_GEOMETRY_GEOMETRY_OBJECT_H_ */

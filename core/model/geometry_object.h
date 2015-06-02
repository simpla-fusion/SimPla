/**
 * @file geometry_object.h
 *
 *  Created on: 2015年6月2日
 *      Author: salmon
 */

#ifndef CORE_MODEL_GEOMETRY_OBJECT_H_
#define CORE_MODEL_GEOMETRY_OBJECT_H_

namespace simpla
{
namespace geometry_object
{
struct Sphere
{
	nTuple<Real, 3> x0;
	Real r;
};
struct Ring;
struct Cylinder;
struct Tetrahedron;
struct Pyramid;
struct Polygons;

template<typename ...>struct ImplicitFunction;

template<>
struct ImplicitFunction<Sphere>
{
	Sphere m_object_;

	ImplicitFunction(nTuple<Real, 3> const & x0, Real radius) :
			m_object_(
			{ x0, radius })
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

}
}  // namespace simpla

#endif /* CORE_MODEL_GEOMETRY_OBJECT_H_ */

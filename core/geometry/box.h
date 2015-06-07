/*
 * box.h
 *
 *  Created on: 2015年6月7日
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_BOX_H_
#define CORE_GEOMETRY_BOX_H_
#include "primitive.h"
namespace simpla
{

namespace geometry
{

namespace model
{

template<typename CS>
struct Box
{
	DEF_NTUPLE_OBJECT(CoordinateSystem, Point<CS> , 2);
};
template<typename OS, typename CoordinateSystem>
OS &operator<<(OS & os, Box<CoordinateSystem> const & geo)
{
	os << geo.as_ntuple();
	return os;
}
}  // namespace model
namespace traits
{
template<typename > struct coordinate_system;
template<typename > struct dimension;
template<typename > struct tag;
template<typename CoordinateSystem>
struct coordinate_system<model::Box<CoordinateSystem>>
{
	typedef CoordinateSystem type;
};

}  // namespace traits
}  // namespace geometry

}  // namespace simpla

#endif /* CORE_GEOMETRY_BOX_H_ */

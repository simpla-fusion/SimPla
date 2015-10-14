/**
 * @file interpolate.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_INTERPOLATE_H
#define SIMPLA_INTERPOLATE_H
namespace simpla
{

template<typename ...> struct Interpolate;
namespace interpolate
{
namespace tags
{
struct linear;
struct spline;
}//namespace tags


}//namespace interpolate
}//namespace simpla
#endif //SIMPLA_INTERPOLATE_H

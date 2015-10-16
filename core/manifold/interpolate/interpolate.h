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

namespace traits
{
template<typename ... T>
struct type_id<Interpolate<T...> >
{
	static std::string name()
	{
		return "Interpolate<" + type_id<T...>::name() + " >";
	}
};


DEFINE_TYPE_ID_NAME(interpolate::tags::linear)
DEFINE_TYPE_ID_NAME(interpolate::tags::spline)

}//namespace traits
}//namespace simpla
#endif //SIMPLA_INTERPOLATE_H

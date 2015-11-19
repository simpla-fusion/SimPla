/**
 * @file diff_scheme.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_CALCULATE_H
#define SIMPLA_CALCULATE_H


#include "../calculus.h"

namespace simpla
{
/**
 * @ingroup diff_geo
 * @{
 *    @defgroup diff_scheme Differential scheme
 * @}
 *
 * @ingroup diff_scheme
 * @{
 */
namespace manifold { namespace policy
{

template<typename...> struct DiffScheme;

namespace diff_scheme
{
namespace tags
{
struct finite_difference;
struct finite_volume;
struct DG;

}//namespace tags
}//namespace diff_scheme
}}
namespace traits
{

template<typename ... T>
struct type_id<manifold::policy::DiffScheme<T...> >
{
    static std::string name()
    {
        return "DiffScheme< \"" + type_id<T...>::name() + "\" >";
    }
};

DEFINE_TYPE_ID_NAME(manifold::policy::diff_scheme::tags::finite_difference)

DEFINE_TYPE_ID_NAME(manifold::policy::diff_scheme::tags::finite_volume)

DEFINE_TYPE_ID_NAME(manifold::policy::diff_scheme::tags::DG)

}//namespace traits
/**  @}*/
}//namespace simpla
#endif //SIMPLA_CALCULATE_H

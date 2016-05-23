/**
 * @file compound_distribution.h
 * @author salmon
 * @date 2015-11-27.
 */

#ifndef SIMPLA_COMPOUND_DISTRIBUTION_H
#define SIMPLA_COMPOUND_DISTRIBUTION_H

namespace simpla
{
template<typename ...Dist>
struct compound_distribution
{
    std::tuple<traits::value_type_t< Dist>...> value_type;

};
}

#endif //SIMPLA_COMPOUND_DISTRIBUTION_H

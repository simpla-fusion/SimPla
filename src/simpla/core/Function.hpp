#ifndef SIMPLA_FUNCTION_HPP
#define SIMPLA_FUNCTION_HPP

#include "Domain.hpp"
#include "Expression.hpp"
namespace simpla {
template <typename... T>
class Function : public Expression<T...> {
    Domain domain;
};

}  // namespace simpla
#endif SIMPLA_FUNCTION_HPP
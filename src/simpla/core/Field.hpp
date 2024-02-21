
#ifndef SIMPLA_FIELD_HPP
#define SIMPLA_FIELD_HPP

#include <type_traits>

#include "Domain.hpp"
#include "Expression.hpp"

namespace simpla {

template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;

template <typename T, Derived<Domain> TDomain, typename... Others>
class Field : public Expression<T...> {
    TDomain domain;
};

}  // namespace simpla

#endif  // SIMPLA_FIELD_HPP
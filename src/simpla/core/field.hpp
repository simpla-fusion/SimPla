/**
 * @file field.hpp
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_HPP
#define SIMPLA_FIELD_HPP
#include <memory>

#include "simpla/core/data_block.hpp"
#include "simpla/core/domain.hpp"
#include "simpla/core/expression.hpp"
#include "simpla/utils/logger.hpp"

namespace simpla {

template <typename TValue, int IFORM = 0, typename TDomain = Domain<1>>
class Field : public Variable<TValue, IFORM> {
   public:
    static constexpr int form = IFORM;
    typedef TValue ValueType;
    typedef TDomain DomainType;

   private:
    typedef Field<TValue, IFORM, TDomain> ThisType;
    typedef Variable<TValue, IFORM> BaseType;

    DomainType const& m_domain_;

    DataBlock<ValueType> m_data_;

   public:
    template <typename... Args>
    explicit Field(DomainType const& domain, Args&&... args)
        : m_domain_(domain),
          m_data_(m_domain_.template create_datablock<ValueType, IFORM>(std::forward<Args>(args)...)) {}

    template <typename TContext, typename... Args>
    explicit Field(TContext* ctx, Args&&... args) : Field(ctx->domain(), std::forward<Args>(args)...) {}

    ~Field() override = default;

    Field(ThisType const& other) = delete;
    Field(ThisType&& other) = delete;

    DomainType const& domain() const { return m_domain_; }

    ThisType& operator=(ThisType const& other) {
        set_value(other.m_data_);
        return *this;
    }

    template <typename TR>
    ThisType& operator=(TR&& rhs) {
        set_value(std::forward<TR>(rhs));
        return *this;
    };

    template <typename... Args>
    decltype(auto) get_value(Args&&... args) const {
        return m_domain_.get_value(m_data_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    ThisType& set_value(Args&&... args) {
        m_domain_.set_value(m_data_, std::forward<Args>(args)...);
        return *this;
    }

};  // class Field

// namespace traits {
// template <typename TM, typename TV, int IFORM, int... DOF>
// struct iform<Field<TV, IFORM, TM>> : public std::integral_constant<int, IFORM> {};
// }  // namespace traits

}  // namespace simpla

#endif  // SIMPLA_FIELD_H

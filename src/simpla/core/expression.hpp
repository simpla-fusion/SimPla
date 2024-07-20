//
// Created by salmon on 2024-02-10.
//

#ifndef SIMPLA_EXPRESSION_H_
#define SIMPLA_EXPRESSION_H_
#include "simpla/core/domain.hpp"

namespace simpla {
template <typename TValue, int IFORM = 0>
class Expression {
   public:
    typedef TValue value_type;
    static constexpr int form = IFORM;

   public:
    template <typename... Args>
    explicit Expression(Args&&... args) {}
    virtual ~Expression() = default;

    Expression(Expression const& other) = delete;
    Expression(Expression&& other) = delete;
};

template <typename TValue, int IFORM = 0>
class Variable : public Expression<TValue, IFORM> {
   public:
    Variable() = default;
    ~Variable() = default;
};

}  // namespace simpla

#endif  // SIMPLA_EXPRESSION_H_

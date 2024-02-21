//
// Created by salmon on 2024-02-10.
//

#ifndef SIMPLA_EXPRESSION_H_
#define SIMPLA_EXPRESSION_H_

namespace simpla {
template <typename...>
class Expression;

template <typename...>
class Variable : public Expression<> {
   public:
    Variable() = default;
    ~Variable() = default;
};

}  // namespace simpla

#endif  // SIMPLA_EXPRESSION_H_

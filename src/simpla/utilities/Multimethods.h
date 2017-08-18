//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_MULTIMETHODS_H
#define SIMPLA_MULTIMETHODS_H

#include <memory>
namespace simpla {
template <typename V>
struct VisitorT;
struct Visitor {
    template <typename U>
    std::shared_ptr<VisitorT<U>> New();
};

template <typename V>
struct VisitorT {
    virtual void visit(V &) = 0;
};

template <typename V>
struct MultiMethods {
    void accept(Visitor &visitor) { visitor.New<V>()->visit(*this); }
};
}  // namespace simpla
#endif  // SIMPLA_MULTIMETHODS_H

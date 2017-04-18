/**
 *  @file observer.h
 */

#ifndef SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H
#define SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

#include <memory>
#include <set>

namespace simpla {
namespace design_pattern {

template <typename SIGNATURE>
struct Observable;
template <typename SIGNATURE>
struct Observer;

template <typename... Args>
struct Observer<void(Args...)> {
    typedef Observable<void(Args...)> observable_type;
    Observer() {}
    virtual ~Observer() {
        for (auto *item : m_subjects_) { Disconnect(item); }
    };

    void Connect(observable_type *p) {
        if (p != nullptr && m_subjects_.emplace(p).second) { p->Attach(this); }
    }
    void Disconnect(observable_type *p) {
        if (p != nullptr && m_subjects_.erase(p) > 0) { p->Detach(this); }
    }

    virtual void OnNotify(Args...) = 0;

   private:
    std::set<observable_type *> m_subjects_;
};

template <typename... Args>
struct Observable<void(Args...)> {
    typedef Observer<void(Args...)> observer_type;
    std::set<observer_type *> m_observers_;

   public:
    Observable() {}
    virtual ~Observable() {}

    void Attach(observer_type *observer) { m_observers_.emplace(observer); };
    void Detach(observer_type *observer) { m_observers_.erase(observer); }
    void Notify(Args... args) {
        for (auto &ob : m_observers_) { ob->OnNotify(args...); }
    }
};
}  // namespace design_pattern {
}  // namespace simpla
#endif  // SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

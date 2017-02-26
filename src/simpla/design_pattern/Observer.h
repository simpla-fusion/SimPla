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
    virtual ~Observer() { Disconnect(); };

    void Connect(observable_type *subject) {
        if (m_subject_ != subject) {
            Disconnect();
            m_subject_ = subject;
            if (m_subject_ != nullptr) { m_subject_->Attach(this); }
        }
    }
    void Disconnect() {
        if (m_subject_ != nullptr) { m_subject_->Detach(this); }
        m_subject_ = nullptr;
    }

    virtual void OnNotify(Args...) = 0;

   private:
    observable_type *m_subject_ = nullptr;
};

template <typename... Args>
struct Observable<void(Args...)> {
    typedef Observer<void(Args...)> observer_type;
    std::set<observer_type *> m_observers_;

   public:
    Observable() {}
    virtual ~Observable() {}

    virtual void Attach(observer_type *observer) { m_observers_.insert(observer); };
    virtual void Detach(observer_type *observer) { m_observers_.erase(observer); }
    virtual void Notify(Args... args) {
        for (auto &ob : m_observers_) { ob->OnNotify(args...); }
    }
};
}  // namespace design_pattern {
}  // namespace simpla
#endif  // SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

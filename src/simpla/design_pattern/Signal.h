//
// Created by salmon on 17-2-26.
//

#ifndef SIMPLA_SIGNAL_H
#define SIMPLA_SIGNAL_H

#include <set>
namespace simpla {
namespace design_pattern {
template <typename... Args>
class Signal {
    typedef Signal<Args...> this_type;
    std::set<this_type*> m_in_;
    std::set<this_type*> m_out_;

   public:
    Signal() {}
    ~Signal() {
        for (this_type* sg : m_in_) { sg->RemoveInput(this); }
        for (this_type* sg : m_out_) { sg->RemoveOutput(this); }
    }
    void SendMessage(Args... args) {
        for (this_type* item : m_out_) { item->ReceiveMessage(this, args...); }
    }
    virtual void ReceiveMessage(this_type*, Args... args) { SendMessage(args...); };

    void AddInput(this_type* s) { m_in_.emplace(s); }
    void RemoveInput(this_type* s) { m_in_.erase(s); }
    void AddOutput(this_type* s) { m_out_.emplace(s); }
    void RemoveOutput(this_type* s) { m_out_.erase(s); }
};
}
}
#endif  // SIMPLA_SIGNAL_H

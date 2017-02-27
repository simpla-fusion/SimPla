//
// Created by salmon on 17-2-26.
//

#ifndef SIMPLA_SIGNAL_H
#define SIMPLA_SIGNAL_H

#include <simpla/engine/SPObject.h>
#include <simpla/mpl/macro.h>
#include <functional>
#include <map>
#include <set>

namespace simpla {
class SPObject;
namespace design_pattern {
template <typename...>
class Signal;

template <typename TRet, typename... Args>
class Signal<TRet(Args...)> {
    typedef Signal<Args...> this_type;
    typedef std::function<TRet(Args...)> call_back_type;
    mutable std::map<int, call_back_type> m_slots_;
    mutable id_type m_count_ = 0;

   public:
    Signal() {}
    ~Signal() {}

    TRet operator()(Args&&... args) const { return emit(std::forward<Args>(args)...); }
    TRet emit(Args&&... args) const {
        for (auto const& item : m_slots_) { item.second(std::forward<Args>(args)...); }
    };
    template <typename TReduction>
    TRet emit(Args&&... args, TReduction reduction) const {
        TRet res;
        for (auto const& item : m_slots_) { res = reduction(res, item.second(std::forward<Args>(args)...)); }
        return res;
    };
    id_type Connect(std::function<TRet(Args...)> const& fun) {
        m_slots_.emplace(m_count_, fun);
        ++m_count_;
        return m_count_;
    }
    template <typename T, TRet (T::*mem_ptr)(Args...)>
    id_type Connect(T* recv) {
//        auto send_id = Connect(recv->*mem_ptr);
        //        auto recv_id = recv->OnDestroy.Connect([=]() { this->Disconnect(send_id); });
        //        OnDestroy.Connect([=]() { recv->OnDestroy.Disconnect(recv_id); });
        return 0;
    }
    void Disconnect(id_type id) { m_slots_.erase(id); }
    //    Signal<void()> OnDestroy;
};

}  // namespace design_pattern {
}  // namespace simpla {

#endif  // SIMPLA_SIGNAL_H

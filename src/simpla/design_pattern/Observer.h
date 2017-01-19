/**
 *  @file observer.h
 */

#ifndef SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H
#define SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

#include <memory>
#include <set>

namespace simpla { namespace design_pattern
{

template<typename SIGNATURE> struct Observable;
template<typename SIGNATURE> struct Observer;

template<typename ...Args>
struct Observer<void(Args...)>
{

    typedef Observable<void(Args...)> observable_type;

    Observer() {}

    virtual ~Observer()
    {
        disconnect();
    };

    virtual void connect(observable_type *subject)
    {
        subject->connect(this);
        m_subject_ = subject;
    }

    void disconnect()
    {
        if (m_subject_ != nullptr) { m_subject_->disconnect(this); }

//        if (m_subject_ != nullptr) { m_subject_->Disconnect(this); }
//        std::shared_ptr<observable_type>(nullptr).swap();
        m_subject_ = nullptr;
    }

    virtual void accept(Args ...) = 0;

private:
    observable_type *m_subject_ = nullptr;

};

template<typename Signature>
struct Observable
{
    typedef Observer<Signature> observer_type;

    std::set<observer_type *> m_observers_;

    Observable() {}

    virtual ~Observable() {}

    template<typename ...Args>
    void accept(Args &&...args) { for (auto &item:m_observers_) { item->accept(std::forward<Args>(args)...); }}


    virtual void connect(observer_type *observer) { m_observers_.insert(observer); };
    virtual void disconnect(observer_type *observer) { m_observers_.erase(observer); }

    template<typename T, typename ...Args>
    typename std::enable_if<std::is_polymorphic<observer_type>::value,
            std::shared_ptr<T>>::type create_observer(Args &&...args)
    {
        auto res = std::make_shared<T>(std::forward<Args>(args)...);

        connect(static_cast<observer_type *>(res.get()));

        return res;

    };



    virtual void foreach(std::function<void(observer_type &)> const &fun)
    {
        for (auto &ob:m_observers_) { fun(*ob); }
    }

    virtual void foreach(std::function<void(observer_type const &)> const &fun) const
    {
        for (auto const &ob:m_observers_) { fun(*ob); }
    }
};


}}// namespace simpla
#endif //SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

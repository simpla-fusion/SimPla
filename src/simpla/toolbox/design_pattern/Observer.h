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

    virtual ~Observer() { disconnect(); };

    void connect(observable_type &subject) { m_subject_ = subject.shared_from_this(); }

    void disconnect()
    {
        if (m_subject_ != nullptr) { m_subject_->disconnect(this); }
        std::shared_ptr<observable_type>(nullptr).swap(m_subject_);
    }

    virtual void notify(Args ...) = 0;

private:
    std::shared_ptr<observable_type> m_subject_;

};

template<typename Signature>
struct Observable : public std::enable_shared_from_this<Observable<Signature>>
{
    typedef Observer<Signature> observer_type;

    std::map<observer_type *, std::shared_ptr<observer_type>> m_observers_;


    Observable() {}

    virtual ~Observable() {}

    template<typename ...Args>
    void notify(Args &&...args) { for (auto &item:m_observers_) { item.second->notify(std::forward<Args>(args)...); }}


    void connect(std::shared_ptr<observer_type> observer)
    {
        observer->connect(*this);
        m_observers_.insert(std::make_pair(observer.get(), observer));
    };

    template<typename T, typename ...Args>
    typename std::enable_if<std::is_polymorphic<observer_type>::value,
            std::shared_ptr<T>>::type create_observer(Args &&...args)
    {
        auto res = std::make_shared<T>(std::forward<Args>(args)...);

        connect(std::dynamic_pointer_cast<observer_type>(res));

        return res;

    };


    void disconnect(observer_type *observer)
    {
        auto it = m_observers_.find(observer);

        if (it != m_observers_.end())
        {
            it->second->disconnect();

            m_observers_.erase(observer);
        }
    }

    void remove(std::shared_ptr<observer_type> &observer) { disconnect(observer.get()); }

    virtual void foreach(std::function<void(observer_type &)> const &fun)
    {
        for (auto &ob:m_observers_) { fun(*ob.second); }
    }

    virtual void foreach(std::function<void(observer_type const &)> const &fun) const
    {
        for (auto const &ob:m_observers_) { fun(*ob.second); }
    }
};


}}// namespace simpla
#endif //SIMPLA_toolbox_DESIGN_PATTERN_OBSERVER_H

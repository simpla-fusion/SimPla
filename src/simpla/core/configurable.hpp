//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <memory>

#include "simpla/config.h"
#include "simpla/core/entry.hpp"
namespace simpla {
namespace data {
struct Configurable;
template <typename TV>
struct PropertyObserverT;
struct PropertyObserver {
    PropertyObserver(Configurable *, std::string const &);
    virtual ~PropertyObserver();

    template <typename TV>
    void set_value(TV const &v) {
        if (auto *p = dynamic_cast<PropertyObserverT<TV> *>(this)) { p->Set(v); }
    }
    template <typename TV>
    TV const &get_value() const {
        auto *p = dynamic_cast<PropertyObserverT<TV> const *>(this);
        ASSERT(p != nullptr);
        return p->Get();
    }
    template <typename U>
    bool CheckValue(U const &u) const {
        bool res = false;
        if (auto const *p = dynamic_cast<PropertyObserverT<U> const *>(this)) { res = p->Get() == u; }
        return res;
    }
    virtual void Pop(std::shared_ptr<data::Entry> const &, std::string const &name) = 0;
    virtual void Push(std::shared_ptr<const data::Entry> const &, std::string const &name) = 0;

    std::string const &GetName() const { return m_name_; }

   private:
    Configurable *m_host_;
    std::string m_name_;
};
template <typename TV>
struct PropertyObserverT : public PropertyObserver {
    typedef TV value_type;
    value_type *m_p_value_;
    PropertyObserverT(Configurable *observer, std::string name, value_type *p)
        : PropertyObserver(observer, name), m_p_value_(p) {}
    ~PropertyObserverT() override = default;

    void Set(value_type const &v) { *m_p_value_ = v; }
    value_type const &Get() const { return *m_p_value_; }

    void Pop(std::shared_ptr<data::Entry> const &cfg, std::string const &name) override {
        cfg->set_value<value_type>(name, *m_p_value_);
    }
    void Push(std::shared_ptr<const data::Entry> const &cfg, std::string const &name) override {
        *m_p_value_ = cfg->get_value<value_type>(name, *m_p_value_);
    };
};
#define SP_PROPERTY(_TYPE_, _NAME_)                                                                         \
    void Set##_NAME_(_TYPE_ const &_v_) { m_##_NAME_##_ = _v_; }                                            \
    _TYPE_ Get##_NAME_() const { return m_##_NAME_##_; }                                                    \
                                                                                                            \
    simpla::data::PropertyObserverT<_TYPE_> m_##_NAME_##_observer_{this, __STRING(_NAME_), &m_##_NAME_##_}; \
    _TYPE_ m_##_NAME_##_

struct Configurable {
   private:
    std::shared_ptr<Entry> m_db_;
    std::map<std::string, PropertyObserver *> m_observers_;

   public:
    Configurable();
    Configurable(Configurable const &other);
    virtual ~Configurable();

    template <typename TV>
    int SetProperty(std::string const &key, TV const &v) {
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            it->second->set_value(v);
        } else {
            m_db_->set_value(key, v);
        }
        return SP_SUCCESS;
    }
    template <typename TV>
    TV GetProperty(std::string const &key) const {
        TV res;
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            res = it->second->get_value<TV>();
        } else {
            res = m_db_->get_value<TV>(key);
        }
        return res;
    }
    template <typename TV>
    TV GetProperty(std::string const &key, TV const &default_value) const {
        TV res;
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            res = it->second->get_value<TV>();
        } else {
            res = m_db_->get_value<TV>(key, default_value);
        }
        return res;
    }

    template <typename... Args>
    void SetProperties(Args &&...args) {
        m_db_->set_value(std::forward<Args>(args)...);
        Push();
    }
    template <typename U>
    bool CheckProperty(std::string const &key, U const &u) const {
        bool res = false;
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            res = it->second->CheckValue(u);
        } else {
            res = m_db_->Check(key, u);
        }
        return res;
    }
    bool CheckProperty(std::string const &key) const { return CheckProperty(key, true); }

    void Detach(PropertyObserver *attr);
    void Attach(PropertyObserver *attr);
    void Push(std::shared_ptr<const Entry> const &cfg = nullptr);
    void Pop(std::shared_ptr<Entry> const &cfg = nullptr);
    void Pop(std::shared_ptr<Entry> const &cfg = nullptr) const;
    void SetDB(std::shared_ptr<const Entry> const &cfg);
    void SetDB(std::shared_ptr<Entry> const &cfg);
    void Link(Configurable *cfg);
    void Link(Configurable const *cfg);
    void Link(std::shared_ptr<Configurable> const &cfg) { Link(cfg.get()); }
    void Link(std::shared_ptr<const Configurable> const &cfg) { Link(cfg.get()); }
};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_CONFIGURABLE_H

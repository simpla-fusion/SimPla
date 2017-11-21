//
// Created by salmon on 17-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include "DataEntry.h"
namespace simpla {
namespace data {
struct Configurable;
template <typename TV>
struct PropertyObserverT;
struct PropertyObserver {
    PropertyObserver(Configurable *, std::string const &);
    virtual ~PropertyObserver();

    template <typename TV>
    void SetValue(TV const &v) {
        if (auto *p = dynamic_cast<PropertyObserverT<TV> *>(this)) { p->Set(v); }
    }
    template <typename TV>
    TV const &GetValue() const {
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
    virtual void Pop(std::shared_ptr<data::DataEntry> const &, std::string const &name) = 0;
    virtual void Push(std::shared_ptr<const data::DataEntry> const &, std::string const &name) = 0;

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

    void Pop(std::shared_ptr<data::DataEntry> const &cfg, std::string const &name) override {
        cfg->SetValue<value_type>(name, *m_p_value_);
    }
    void Push(std::shared_ptr<const data::DataEntry> const &cfg, std::string const &name) override {
        *m_p_value_ = cfg->GetValue<value_type>(name, *m_p_value_);
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
    std::shared_ptr<DataEntry> m_db_;
    std::map<std::string, PropertyObserver *> m_observers_;

   public:
    Configurable();
    Configurable(Configurable const &other);
    virtual ~Configurable();

    template <typename TV>
    int SetProperty(std::string const &key, TV const &v) {
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            it->second->SetValue(v);
        } else {
            m_db_->SetValue(key, v);
        }
        return SP_SUCCESS;
    }
    template <typename TV>
    TV GetProperty(std::string const &key) const {
        TV res;
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            res = it->second->GetValue<TV>();
        } else {
            res = m_db_->GetValue<TV>(key);
        }
        return res;
    }
    template <typename TV>
    TV GetProperty(std::string const &key, TV const &default_value) const {
        TV res;
        auto it = m_observers_.find(key);
        if (it != m_observers_.end()) {
            res = it->second->GetValue<TV>();
        } else {
            res = m_db_->GetValue<TV>(key, default_value);
        }
        return res;
    }

    template <typename... Args>
    void SetProperties(Args &&... args) {
        m_db_->SetValue(std::forward<Args>(args)...);
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
    void Push(std::shared_ptr<const DataEntry> const &cfg = nullptr);
    void Pop(std::shared_ptr<DataEntry> const &cfg = nullptr);
    void Pop(std::shared_ptr<DataEntry> const &cfg = nullptr) const;
    void SetDB(std::shared_ptr<const DataEntry> const &cfg);
    void SetDB(std::shared_ptr<DataEntry> const &cfg);
    void Link(Configurable *cfg);
    void Link(Configurable const *cfg);
    void Link(std::shared_ptr<Configurable> const &cfg) { Link(cfg.get()); }
    void Link(std::shared_ptr<const Configurable> const &cfg) { Link(cfg.get()); }
};

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_CONFIGURABLE_H

//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATREE_H_
#define SIMPLA_DATATREE_H_

#include <simpla/SIMPLA_config.h>

#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>

#include "DataEntity.h"
#include "HeavyData.h"
#include "LightData.h"

namespace simpla {
namespace data {
class KeyValue;
template <>
struct entity_traits<std::initializer_list<KeyValue>> {
    typedef int_const<DataEntity::TABLE> type;
};

template <typename U>
std::shared_ptr<DataEntity> make_shared_entity(U const& c, ENABLE_IF(entity_traits<std::decay_t<U>>::type::value ==
                                                                     DataEntity::TABLE)) {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(c));
}

class KeyValue {
   private:
    std::string m_key_;
    std::shared_ptr<DataEntity> m_value_;

   public:
    KeyValue(char const* k) : m_key_(k), m_value_(make_shared_entity(true)) {}
    KeyValue(std::string const& k) : m_key_(k), m_value_(make_shared_entity(true)) {}
    template <typename U>
    KeyValue(std::string const& k, U const& u) : m_key_(k), m_value_(make_shared_entity(u)) {}
    KeyValue(KeyValue const& other) : m_key_(other.m_key_), m_value_(other.m_value_) {}
    KeyValue(KeyValue&& other) : m_key_(other.m_key_), m_value_(other.m_value_) {}
    ~KeyValue() {}

    template <typename U>
    KeyValue& operator=(U const& u) {
        m_value_ = make_shared_entity(u);
        return *this;
    }

    KeyValue& operator=(char const* c) {
        m_value_ = make_shared_entity(std::string(c));
        return *this;
    }
    KeyValue& operator=(char* c) {
        m_value_ = make_shared_entity(std::string(c));
        return *this;
    }
    KeyValue& operator=(std::initializer_list<KeyValue> const& u) {
        m_value_ = make_shared_entity(u);
        return *this;
    }

    std::string const& key() const { return m_key_; }
    std::shared_ptr<DataEntity> const& value() const { return m_value_; }
};

inline KeyValue operator"" _(const char* c, std::size_t n) {
    return KeyValue{std::string(c), make_shared_entity(true)};
}

/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);

   public:
    DataTable();

    DataTable(std::initializer_list<KeyValue> const&);

    template <typename U>
    DataTable(std::string const& key, U const& v) : DataTable() {
        setValue(key, v);
    };

    DataTable(std::string const& key, std::shared_ptr<DataEntity> const& v) : DataTable() { setValue(key, v); };

    //    DataTable(DataTable const&);

    DataTable(DataTable&&);

    virtual ~DataTable();

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;

    virtual bool isTable() const { return true; };

    virtual bool empty() const;

    virtual bool has(std::string const& key) const;

    virtual void foreach (std::function<void(std::string const& key, DataEntity const&)> const&) const;

    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun);

    template <typename T>
    bool check(std::string const& url, T const& v) const {
        DataEntity const* p = find(url);
        return p != nullptr && p->asLight().equal(v);
    };

    virtual DataEntity const* find(std::string const& url) const;

    virtual void Parse(){};

    virtual void Parse(std::string const& str);

    template <int N>
    void Parse(const char* c) {
        Parse(std::string(c));
    };

    template <typename U>
    void Parse(std::pair<std::string, U> const& k_v) {
        setValue(k_v.first, k_v.second);
    };

    template <typename T0, typename T1, typename... Args>
    void Parse(T0 const& a0, T1 const& a1, Args&&... args) {
        Parse(a0);
        Parse(a1, std::forward<Args>(args)...);
    };

    void insert(KeyValue const& k_v) { setValue(k_v.key(), k_v.value()); };

    void insert(){};

    template <typename... Others>
    void insert(KeyValue const& k_v, Others&&... others) {
        setValue(k_v.key(), k_v.value());
        insert(std::forward<Others>(others)...);
    };

    void insert(std::initializer_list<KeyValue> const& other);

    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent
     * '''table''' as needed.
     */

    virtual std::shared_ptr<DataEntity> setValue(std::string const& key, std::shared_ptr<DataEntity> const& v);

    template <typename U>
    auto setValue(std::string const& url, U const& v) {
        return setValue(url, make_shared_entity(v));
    }

    virtual DataTable* CreateTable(std::string const& url);

    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> Get(std::string const& url);

    template <typename U>
    U getValue(std::string const& url) const {
        return at(url).as<U>();
    }

    template <typename U>
    U getValue(std::string const& url, U const& u) const {
        auto p = find(url);
        return p == nullptr ? u : p->asLight().template as<U>();
    }
    std::string getValue(std::string const& url, char const* u) const {
        auto p = find(url);
        return p == nullptr ? std::string(u) : p->asLight().template as<std::string>();
    }
    //    template<typename U> U const &getValue(std::string const &url, U const &u)
    //    {
    //        auto *p = find(url);
    //
    //        if (p != nullptr) { return p->as<U>(); } else { return setValue(url, u)->as<U>();
    //        }
    //    }

    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to
     * key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    virtual DataEntity& at(std::string const& key);

    virtual DataEntity const& at(std::string const& key) const;

    LightData& asLight(std::string const& url) { return at(url).asLight(); };

    LightData const& asLight(std::string const& url) const { return at(url).asLight(); };

    HeavyData& asHeavy(std::string const& url) { return at(url).asHeavy(); };

    HeavyData const& asHeavy(std::string const& url) const { return at(url).asHeavy(); };

    DataTable& asTable(std::string const& url) { return at(url).asTable(); };

    DataTable const& asTable(std::string const& url) const { return at(url).asTable(); };

    template <typename U>
    U& as(std::string const& url) {
        return at(url).asLight().template as<U>();
    }
    template <typename U>
    U const& as(std::string const& url) const {
        return at(url).asLight().template as<U>();
    }

   protected:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

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
std::shared_ptr<DataEntity> make_shared_entity(
    U const& c, ENABLE_IF(entity_traits<std::decay_t<U>>::type::value == DataEntity::TABLE)) {
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
        set_value(key, v);
    };

    DataTable(std::string const& key, std::shared_ptr<DataEntity> const& v) : DataTable() {
        set_value(key, v);
    };

    //    DataTable(DataTable const&);

    DataTable(DataTable&&);

    virtual ~DataTable();

    virtual std::ostream& print(std::ostream& os, int indent = 0) const;

    virtual bool is_table() const { return true; };

    virtual bool empty() const;

    virtual bool has(std::string const& key) const;

    virtual void foreach (
        std::function<void(std::string const& key, DataEntity const&)> const&) const;

    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun);

    template <typename T>
    bool check(std::string const& url, T const& v) const {
        DataEntity const* p = find(url);
        return p != nullptr && p->as_light().equal(v);
    };

    virtual DataEntity const* find(std::string const& url) const;

    virtual void parse(){};

    virtual void parse(std::string const& str);

    template <int N>
    void parse(char const c[N]) {
        parse(std::string(c));
    };

    template <typename U>
    void parse(std::pair<std::string, U> const& k_v) {
        set_value(k_v.first, k_v.second);
    };

    template <typename T0, typename T1, typename... Args>
    void parse(T0 const& a0, T1 const& a1, Args&&... args) {
        parse(a0);
        parse(a1, std::forward<Args>(args)...);
    };

    void insert(KeyValue const& k_v) { set_value(k_v.key(), k_v.value()); };

    void insert(){};

    template <typename... Others>
    void insert(KeyValue const& k_v, Others&&... others) {
        set_value(k_v.key(), k_v.value());
        insert(std::forward<Others>(others)...);
    };

    void insert(std::initializer_list<KeyValue> const& other);

    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent
     * '''table''' as needed.
     */

    virtual std::shared_ptr<DataEntity> set_value(std::string const& key,
                                                  std::shared_ptr<DataEntity> const& v);

    template <typename U>
    auto set_value(std::string const& url, U const& v) {
        return set_value(url, make_shared_entity(v));
    }

    virtual DataTable* create_table(std::string const& url);

    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> get(std::string const& url);

    template <typename U>
    U const& get_value(std::string const& url) const {
        return at(url).as<U>();
    }

    template <typename U>
    U const& get_value(std::string const& url, U const& u) const {
        auto p = find(url);
        return p == nullptr ? u : p->as_light().template as<U>();
    }

    //    template<typename U> U const &get_value(std::string const &url, U const &u)
    //    {
    //        auto *p = find(url);
    //
    //        if (p != nullptr) { return p->as<U>(); } else { return set_value(url, u)->as<U>();
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

    LightData& as_light(std::string const& url) { return at(url).as_light(); };

    LightData const& as_light(std::string const& url) const { return at(url).as_light(); };

    HeavyData& as_heavy(std::string const& url) { return at(url).as_heavy(); };

    HeavyData const& as_heavy(std::string const& url) const { return at(url).as_heavy(); };

    DataTable& as_table(std::string const& url) { return at(url).as_table(); };

    DataTable const& as_table(std::string const& url) const { return at(url).as_table(); };

    template <typename U>
    U& as(std::string const& url) {
        return at(url).as_light().template as<U>();
    }
    template <typename U>
    U const& as(std::string const& url) const {
        return at(url).as_light().template as<U>();
    }

   protected:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATREE_H_
#define SIMPLA_DATATREE_H_

#include <simpla/SIMPLA_config.h>
//#include <simpla/engine/SPObjectHead.h>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include "DataEntity.h"

namespace simpla {
namespace data {

class DataTable;

class KeyValue {
   public:
    KeyValue(unsigned long long int n, std::shared_ptr<DataEntity> const& p);
    KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p);
    KeyValue(KeyValue const& other);
    KeyValue(KeyValue&& other);
    ~KeyValue();

    //    KeyValue& operator=(std::initializer_list<KeyValue> const& u);
    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        SetValue(make_data_entity(u));
        return *this;
    }

    void SetValue(std::shared_ptr<DataEntity> const&);

    std::string const& key() const;
    DataEntity const& value() const;
    DataEntity const* pointer() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data_entity(true)}; }
inline KeyValue operator"" _(unsigned long long int n) { return KeyValue{n, make_data_entity(0)}; }

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u);
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 */
class DataTable : public DataEntity {
    SP_OBJECT_BASE(DataTable);

   public:
    DataTable();
    DataTable(const std::initializer_list<KeyValue>& c);
    DataTable(DataTable const&);
    DataTable(DataTable&&);
    virtual ~DataTable();
    virtual void Parse(std::string const& str);
    virtual void ParseFile(std::string const& str);
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isTable() const { return true; };
    virtual bool empty() const;
    virtual bool has(std::string const& key) const;

    virtual std::type_info const& type() const { return typeid(DataTable); };

    virtual KeyValue* find(std::string const& url) const;

    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent
     * '''table''' as needed.
     */
    virtual KeyValue* Set(KeyValue const& v, std::string const& prefix = "");

    void Set(std::initializer_list<KeyValue> const& c);

    template <typename... Others>
    void Set(KeyValue const& k_v, KeyValue const& second, Others&&... others) {
        Set(k_v);
        Set(second, std::forward<Others>(others)...);
    }

    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual KeyValue* Get(std::string const& url);
    virtual KeyValue const* Get(std::string const& url) const;
    virtual KeyValue* CreateTable(std::string const& url);

    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to
     * key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    KeyValue& at(std::string const& url) { return *Get(url); };
    KeyValue const& at(std::string const& url) const { return *Get(url); };

    //    template <typename U>
    //    void SetValue(std::string const& url, U const& v) {
    //        Set(url, make_shared_entity(v));
    //    }
    //
    //    template <typename U>
    //    void SetValue(std::pair<std::string, U> const& k_v) {
    //        SetValue(k_v.first, k_v.second);
    //    };
    //    template <typename U, typename URL>
    //    U& GetValue(URL const& url) {
    //        return Get(url)->GetValue<U>();
    //    }
    //    template <typename U, typename URL>
    //    U const& GetValue(URL const& url) const {
    //        return Get(url)->GetValue<U>();
    //    }
    //    template <typename U, typename URL>
    //    U& GetValue(URL const& url, U& u) {
    //        auto p = this->Get(url);
    //        return p == nullptr ? u : p->GetValue<U>();
    //    }
    //    template <typename U, typename URL>
    //    U const& GetValue(URL const& url, U const& u) const {
    //        auto p = this->Get(url);
    //        return p == nullptr ? u : p->GetValue<U>();
    //    }
    //    template <typename U>
    //    bool Check(std::string const& url, U const& v) const {
    //        auto p = this->Get(url);
    //        return p != nullptr && p->Check<U>(v);
    //    };
    //    template <typename URL>
    //    DataTable& GetTable(URL const& url) {
    //        return *Get(url)->as<DataTable>();
    //    }
    //    template <typename URL>
    //    DataTable const& GetTable(URL const& url) const {
    //        return *Get(url)->as<DataTable>();
    //    }

   protected:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATREE_H_
#define SIMPLA_DATATREE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObjectHead.h>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include "DataEntity.h"
#include "KeyValue.h"
namespace simpla {
namespace data {

/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);

   public:
    DataTable();
    DataTable(const std::initializer_list<simpla::data::KeyValue>& c);
    // DataTable(DataTable const&);
    DataTable(DataTable&&);
    virtual ~DataTable();
    virtual void Parse(std::string const& str);
    virtual void ParseFile(std::string const& str);
    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual bool isTable() const { return true; };
    virtual bool empty() const;
    virtual bool has(std::string const& key) const;
    //    virtual void foreach (std::function<void(std::string const& key, DataEntity const&)> const&) const;
    //    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun);
    virtual std::shared_ptr<DataEntity> find(std::string const& url) const;

    virtual void Merge(DataTable const&);

    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent
     * '''table''' as needed.
     */
    virtual void Set(std::string const& key, std::shared_ptr<DataEntity> const& v);
    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> Get(std::string const& url);
    virtual std::shared_ptr<DataEntity> Get(std::string const& url) const;
    virtual std::shared_ptr<DataTable> CreateTable(std::string const& url);

    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to
     * key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    DataEntity& at(std::string const& url) { return *Get(url); };
    DataEntity const& at(std::string const& url) const { return *Get(url); };

    void SetValue(KeyValue const& k_v);
    void SetValue(std::initializer_list<KeyValue> const& c);

    template <typename U>
    void SetValue(std::string const& url, U const& v) {
        Set(url, make_shared_entity(v));
    }

    template <typename U>
    void SetValue(std::pair<std::string, U> const& k_v) {
        SetValue(k_v.first, k_v.second);
    };

    template <typename... Others>
    void SetValue(KeyValue const& k_v, KeyValue const& second, Others&&... others) {
        SetValue(k_v);
        SetValue(second, std::forward<Others>(others)...);
    }

    template <typename U, typename URL>
    U& GetValue(URL const& url) {
        return Get(url)->GetValue<U>();
    }
    template <typename U, typename URL>
    U const& GetValue(URL const& url) const {
        return Get(url)->GetValue<U>();
    }
    template <typename U, typename URL>
    U& GetValue(URL const& url, U& u) {
        auto p = this->Get(url);
        return p == nullptr ? u : p->GetValue<U>();
    }
    template <typename U, typename URL>
    U const& GetValue(URL const& url, U const& u) const {
        auto p = this->Get(url);
        return p == nullptr ? u : p->GetValue<U>();
    }

    template <typename U>
    bool Check(std::string const& url, U const& v) {
        auto p = this->Get(url);
        return p != nullptr && p->GetValue<U>() == v;
    };
    template <typename URL>
    DataTable& GetTable(URL const& url) {
        return *Get(url)->as<DataTable>();
    }
    template <typename URL>
    DataTable const& GetTable(URL const& url) const {
        return *Get(url)->as<DataTable>();
    }

   protected:
    struct pimpl_s;
    std::shared_ptr<pimpl_s> m_pimpl_;
};
template <typename U>
class DataTableAdapter : public DataTable, public U {
   public:
    DataTableAdapter() {}
    ~DataTableAdapter() {}

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const { U::Print(os, indent); };
    virtual bool isTable() const { return true; };
    virtual bool empty() const { return U::empty(); };
    virtual bool has(std::string const& key) const { return U::has(key); }
    virtual void foreach (std::function<void(std::string const& key, DataEntity const&)> const&) const {
        UNIMPLEMENTED;
    }
    virtual void foreach (std::function<void(std::string const& key, DataEntity&)> const& fun) { UNIMPLEMENTED; }
    virtual std::shared_ptr<DataEntity> find(std::string const& url) const { UNIMPLEMENTED; }
    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent
     * '''table''' as needed.
     */

    virtual void Set(std::string const& key, std::shared_ptr<DataEntity> const& v) { U::Set(key, v); };
    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> Get(std::string const& url) { UNIMPLEMENTED; }
    virtual std::shared_ptr<DataEntity> Get(std::string const& url) const { UNIMPLEMENTED; };
    virtual DataTable& GetTable(std::string const& url) { return Get(url + ".")->asTable(); }
    virtual const DataTable& GetTable(std::string const& url) const { return Get(url + ".")->asTable(); }
};
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

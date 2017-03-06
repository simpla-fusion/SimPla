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
class DataBackend;

class KeyValue : public std::pair<std::string const, std::shared_ptr<DataEntity>> {
    typedef std::pair<std::string const, std::shared_ptr<DataEntity>> base_type;

   public:
    KeyValue(unsigned long long int n, std::shared_ptr<DataEntity> const& p);
    KeyValue(std::string const& k, std::shared_ptr<DataEntity> const& p);
    KeyValue(KeyValue const& other);
    KeyValue(KeyValue&& other);
    ~KeyValue();

    KeyValue& operator=(KeyValue const& other) {
        //        base_type::operator=(other);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(U const& u) {
        second = make_data_entity(u);
        return *this;
    }

    template <typename U>
    KeyValue& operator=(std::initializer_list<U> const& u) {
        second = make_data_entity(u);
        return *this;
    }
};

inline KeyValue operator"" _(const char* c, std::size_t n) { return KeyValue{std::string(c), make_data_entity(true)}; }
inline KeyValue operator"" _(unsigned long long int n) { return KeyValue{n, make_data_entity(0)}; }

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u);
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 * @design_pattern
 *  - Proxy of DataBeckend
 */
class DataTable : public DataEntity {
    SP_OBJECT_BASE(DataTable);

   public:
    DataTable(DataBackend* p = nullptr);
    DataTable(DataTable const&);
    DataTable(DataTable&&);
    ~DataTable();

    void swap(DataTable& other);
    DataTable& operator=(DataTable const& other) {
        DataTable(other).swap(*this);
        return *this;
    }
    std::shared_ptr<DataBackend>& backend() { return m_backend_; }
    std::shared_ptr<DataBackend> const& backend() const { return m_backend_; }

    bool isTable() const { return true; };
    std::type_info const& type() const { return typeid(DataTable); };
    std::type_info const& backend_type() const;

    std::ostream& Print(std::ostream& os, int indent = 0) const;

    void Parse(std::string const& str);
    void Open(std::string const& url, std::string const& status = "");
    void Flush();
    void Close();

    bool empty() const;
    void clear();
    void reset();

    DataTable* CreateTable(std::string const& url);

    bool Erase(std::string const& k);
    /**
     *  set entity value to '''url'''
     *  insert or assign
     *  If a key equivalent to k already exists in the table, assigns v to the mapped_type corresponding to the key k.
     *  If the key does not exist, inserts the new value as if by insert,
     * @param url
     * @return Returns a pointer to modified/inserted entity, create parent '''table''' as needed.
     */
    DataEntity* Set(std::string const& k, std::shared_ptr<DataEntity> const& v);

    /**
     * @param url
     * @return Returns a pointer to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, returns nullptr
     */
    DataEntity* Get(std::string const& url);
    DataEntity const* Get(std::string const& url) const;

    DataEntity const* find(std::string const& url) const { return Get(url); };

    void Set(KeyValue const& c) { Set(c.first, c.second); };
    void Set(std::initializer_list<KeyValue> const& l) {
        for (auto const& a : l) { Set(a); }
    }
    template <typename... Others>
    void Set(KeyValue const& k_v, KeyValue const& second, Others&&... others) {
        Set(k_v);
        Set(second, std::forward<Others>(others)...);
    }
    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to
     * key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    DataEntity const& at(std::string const& url) const {
        auto p = Get(url);
        if (p == nullptr) { RUNTIME_ERROR << "Can not find  entity [ url : " << url << "]" << std::endl; }
        return *p;
    }
    DataEntity& at(std::string const& url) {
        auto p = Get(url);
        if (p == nullptr) { RUNTIME_ERROR << "Can not find  entity [ url : " << url << "]" << std::endl; }
        return *p;
    };

    DataTable& GetTable(std::string const& url) {
        DataEntity* p = Get(url);
        if (p == nullptr) {
            p = CreateTable(url);
        } else if (!p->isTable()) {
            RUNTIME_ERROR << "illegal convert! url:[" << url << "] is not a table !";
        }
        return *static_cast<DataTable*>(p);
    };
    DataTable const& GetTable(std::string const& url) const {
        DataEntity const* p = Get(url);
        if (p == nullptr || !p->isTable()) { RUNTIME_ERROR << " url:[" << url << "] is not found or not a table !"; }
        return *static_cast<DataTable const*>(p);
    };

    template <typename U>
    U GetValue(std::string const& url) const {
        return at(url).GetValue<U>();
    }
    template <typename U>
    U GetValue(std::string const& url, U const& default_value) const {
        auto p = find(url);
        return p == nullptr ? default_value : p->GetValue<U>();
    }
    template <typename U>
    U GetValue(std::string const& url, U const& default_value) {
        DataEntity* p = Get(url);
        if (p == nullptr) { p = SetValue(url, default_value); }
        return p->GetValue<U>();
    }
    template <typename U>
    DataEntity* SetValue(std::string const& url, U const& v) {
        return Set(url, make_data_entity(v));
    }

    template <typename U>
    void SetValue(std::string const& url, std::initializer_list<U> v) {
        Set(url, make_data_entity(v));
    }

   protected:
    std::shared_ptr<DataBackend> m_backend_;
};

std::shared_ptr<DataEntity> make_data_entity(std::initializer_list<KeyValue> const& u);

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

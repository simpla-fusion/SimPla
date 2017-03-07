//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATREE_H_
#define SIMPLA_DATATREE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObjectHead.h>
#include "DataEntity.h"

namespace simpla {
namespace data {

class DataTable;
class DataBackend;

class KeyValue : public std::pair<std::string const, DataEntity> {
    typedef std::pair<std::string const, DataEntity> base_type;

   public:
    KeyValue(unsigned long long int n, DataEntity const& p);
    KeyValue(std::string const& k, DataEntity const& p);
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

DataEntity make_data_entity(std::initializer_list<KeyValue> const& u);

/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 * @design_pattern
 *  - Proxy of DataBeckend
 */
class DataTable : public DataHolderBase {
    SP_OBJECT_HEAD(DataTable, DataHolderBase);
    DataBackend* m_backend_ = nullptr;

   public:
    DataTable(DataBackend* p = nullptr);
    DataTable(std::string const& url, std::string const& status = "");
    DataTable(DataTable const&);
    DataTable(DataTable&&);
    ~DataTable();
    std::type_info const& type() { return typeid(DataTable); };
    DataHolderBase* Copy() const;
    bool empty() const;
    void swap(DataTable& other);
    DataTable& operator=(DataTable const& other);
    DataBackend* backend() { return m_backend_; }
    DataBackend const* backend() const { return m_backend_; }
    std::ostream& Print(std::ostream& os, int indent = 0) const;
    void Parse(std::string const& str);
    void Open(std::string const& url, std::string const& status = "");
    void Flush();
    void Close();
    void Clear();
    void Reset();
    void Update();
    bool Erase(std::string const& k);
    std::pair<DataEntity*, bool> Insert(std::string const& k);
    std::pair<DataEntity*, bool> Insert(std::string const& k, DataEntity const& v, bool assign_is_exists = true);
    DataEntity* Find(std::string const& url) const;

    /**
     * @param url
     * @return Returns a pointer to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, returns nullptr
     */
    DataEntity& operator[](std::string const& url);
    DataEntity const& operator[](std::string const& url) const;

    //    void SetValue(KeyValue const& c);
    //    void SetValue(std::initializer_list<KeyValue> const& l);
    //    template <typename... Others>
    //    void SetValue(KeyValue const& k_v, KeyValue const& second, Others&&... others) {
    //        SetValue(k_v);
    //        SetValue(second, std::forward<Others>(others)...);
    //    }
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

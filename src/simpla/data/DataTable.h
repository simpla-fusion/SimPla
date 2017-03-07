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
    KeyValue(std::string const& k, DataEntity&& p);

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
    SP_OBJECT_BASE(DataTable);
    DataBackend* m_backend_ = nullptr;

   public:
    DataTable(DataBackend* p = nullptr);
    DataTable(std::string const& url, std::string const& status = "");
    DataTable(DataTable const&);
    DataTable(DataTable&&);
    ~DataTable();
    std::type_info const& type() const { return typeid(DataTable); };

    DataTable* Copy() const;
    bool Update();

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

    /**
     *  PUT and POST are both unsafe methods. However, PUT is idempotent, while POST is not.
     *
     *  HTTP/1.1 SPEC
     *  @quota
     *   The POST method is used to request that the origin server accept the entity enclosed in
     *   the request as a new subordinate of the resource identified by the Request-URI in the Request-Line
     *
     *  @quota
     *  The PUT method requests that the enclosed entity be stored under the supplied Request-URI.
     *  If the Request-URI refers to an already existing resource, the enclosed entity SHOULD be considered as a
     *  modified version of the one residing on the origin server. If the Request-URI does not point to an existing
     *  resource, and that URI is capable of being defined as a new resource by the requesting user agent, the origin
     *  server can create the resource with that URI."
     *
     */

    DataEntity Get(std::string const& uri);
    bool Put(std::string const& uri, DataEntity&& v);
    bool Post(std::string const& uri, DataEntity&& v);
    size_type Delete(std::string const& uri);
    size_type Count(std::string const& uri) const;

    template <typename U>
    bool Put(std::string const& uri, U const& v) {
        return Put(uri, DataEntity{v});
    };

    template <typename U>
    bool Put(std::string const& uri, U&& v) {
        return Put(uri, DataEntity{new DataHolder<U>(std::forward<U>(v))});
    };

    template <typename U>
    bool Post(std::string const& uri, U const& v) {
        return Post(uri, DataEntity{v});
    };

    template <typename U>
    bool Post(std::string const& uri, U&& v) {
        return Put(uri, DataEntity{v});
    };
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATREE_H_

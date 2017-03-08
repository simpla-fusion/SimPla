//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObjectHead.h>
#include <memory>
#include "DataEntity.h"
#include "DataTraits.h"
namespace simpla {
namespace data {

class DataBackend;
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
 * in HDF5, but all node/table are DataEntity.
 * @design_pattern
 *  - Proxy of DataBeckend
 */
class DataTable : public DataEntity {
    SP_OBJECT_HEAD(DataTable, DataEntity);
    std::shared_ptr<DataBackend> m_backend_ = nullptr;

   public:
    DataTable(std::shared_ptr<DataBackend> const& p = nullptr);
    DataTable(std::string const& url, std::string const& status = "");
    DataTable(std::initializer_list<KeyValue> const&);
    DataTable(const DataTable&);
    DataTable(DataTable&&);

    virtual ~DataTable();
    virtual bool isTable() const { return true; }
    std::type_info const& type() const { return typeid(DataTable); };
    std::shared_ptr<DataTable> Copy() const;

    std::shared_ptr<DataBackend> backend() const { return m_backend_; }

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual void Parse(std::string const& str);
    virtual void Open(std::string const& url, std::string const& status = "");
    virtual void Flush();
    virtual void Close();
    virtual void Clear();
    virtual void Reset();

    virtual bool empty() const;
    virtual size_type count() const;

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

    virtual std::shared_ptr<DataEntity> Get(std::string const& key) const;
    virtual bool Set(DataTable const& other);
    virtual bool Set(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual bool Set(std::shared_ptr<DataEntity> const& v);
    virtual bool Add(std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& key);

    //    bool Set(KeyValue const& v) { return Set(v.first, v.second); };
    //    template <typename... Args>
    //    void Set(KeyValue const& v, Args&&... args) {
    //        Set(v.first, v.second);
    //        Set(std::forward<Args>(args)...);
    //    };
    //    void Set(std::initializer_list<KeyValue> const& v) {
    //        for (auto const& item : v) { Set(item.first, item.second); }
    //    };

    template <typename U>
    bool Set(std::string const& uri, U const& v) {
        return Set(uri, make_data_entity(v));
    };

    //    template <typename U>
    //    bool Set(std::string const& uri, U&& v) {
    //        return Set(uri, DataEntity{new DataHolder<U>(std::forward<U>(v))});
    //    };

    template <typename U>
    bool Add(std::string const& uri, U const& v) {
        return Add(uri, DataEntity{v});
    };

    template <typename U>
    bool Add(std::string const& uri, U&& v) {
        return Set(uri, DataEntity{v});
    };

    template <typename U>
    bool Add(U const& u) {
        return Add(make_data_entity(u));
    };

    template <typename U>
    bool Set(std::initializer_list<U> const& u) {
        return Add(make_data_entity(u));
    };
};

}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_

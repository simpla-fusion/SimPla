//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATABACKENDHDF5_H
#define SIMPLA_DATABACKENDHDF5_H
#include "../DataBackend.h"

namespace simpla {
namespace data {
class DataBackendHDF5 : public DataBackend {
   public:
    static constexpr char ext[] = "h5";

    DataBackendHDF5();
    DataBackendHDF5(DataBackendHDF5 const&);
    DataBackendHDF5(DataBackendHDF5&&);

    DataBackendHDF5(std::string const& uri, std::string const& status = "");
    virtual ~DataBackendHDF5();

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual std::unique_ptr<DataBackend> CreateNew() const;
    virtual bool isNull() const;
    virtual void Flush();

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    virtual std::shared_ptr<DataEntity> Get(id_type key) const;
    virtual bool Set(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual bool Set(id_type key, std::shared_ptr<DataEntity> const&);
    virtual bool Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual bool Add(id_type key, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type Delete(id_type key);
    virtual void DeleteAll();
    virtual size_type size() const;
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;
    virtual size_type Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATABACKENDHDF5_H
#define SIMPLA_DATABACKENDHDF5_H
#include "../DataBackend.h"

namespace simpla {
namespace data {
class DataBackendHDF5 : public DataBackend {
    SP_OBJECT_HEAD(DataBackendHDF5, DataBackend);

   public:
    DataBackendHDF5();
    DataBackendHDF5(DataBackendHDF5 const&);
    DataBackendHDF5(DataBackendHDF5&&);
    DataBackendHDF5(std::string const& uri, std::string const& status = "");

    virtual ~DataBackendHDF5();

    virtual void Connect(std::string const& path, std::string const& param = "");
    virtual void Disconnect();

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual std::shared_ptr<DataBackend> Duplicate() const;
    virtual std::shared_ptr<DataBackend> CreateNew() const;

    virtual bool isNull() const;
    virtual void Flush();

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
    virtual void Set(std::string const& URI, std::shared_ptr<DataEntity> const&, bool overwrite = true);
    virtual void Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type size() const;
    virtual size_type ForEach(std::function<void(std::string const &, std::shared_ptr<DataEntity>)> const &) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

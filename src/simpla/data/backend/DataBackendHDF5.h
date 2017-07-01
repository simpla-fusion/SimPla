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
    DataBackendHDF5(DataBackendHDF5&&) noexcept;
    explicit DataBackendHDF5(std::string const& uri, std::string const& status = "");
    ~DataBackendHDF5() override;

    DECLARE_REGISTER_NAME(hdf5)

    bool isNull() const;

    void Connect(std::string const& authority, std::string const& path, std::string const& query = "",
                 std::string const& fragment = "") override;

    void Disconnect() override;

    std::shared_ptr<DataBackend> Duplicate() const override;
    std::shared_ptr<DataBackend> CreateNew() const override;

    void Flush() override;

    std::shared_ptr<DataEntity> Get(std::string const& URI) const override;
    void Set(std::string const& URI, std::shared_ptr<DataEntity> const&, bool overwrite = true) override;
    void Add(std::string const& URI, std::shared_ptr<DataEntity> const&) override;
    void Delete(std::string const& URI) override;
    size_type size() const override;
    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const override;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

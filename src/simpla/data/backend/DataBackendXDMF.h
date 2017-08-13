//
// Created by salmon on 17-8-13.
//

#ifndef SIMPLA_DATABACKENDXDMF_H
#define SIMPLA_DATABACKENDXDMF_H

#include "../DataBackend.h"
namespace simpla {
namespace data {

class DataBackendXDMF : public DataBackend {
    SP_OBJECT_HEAD(DataBackendXDMF, DataBackend);

   public:
    DataBackendXDMF();
    ~DataBackendXDMF() override;
    explicit DataBackendXDMF(std::string const& uri, std::string const& status = "");

    DataBackendXDMF(DataBackendXDMF const&) = delete;
    DataBackendXDMF(DataBackendXDMF&&) noexcept = delete;
    DataBackendXDMF& operator=(DataBackendXDMF const&) = delete;
    DataBackendXDMF& operator=(DataBackendXDMF&&) noexcept = delete;

    bool isNull() const;

    void Connect(std::string const& authority, std::string const& path, std::string const& query = "",
                 std::string const& fragment = "") override;

    void Disconnect() override;

    std::shared_ptr<DataBackend> Duplicate() const override;
    std::shared_ptr<DataBackend> CreateNew() const override;

    void Flush() override;
    std::shared_ptr<DataEntity> Get(std::string const& URI) const override;
    void Set(std::string const& URI, const std::shared_ptr<DataEntity>&) override;
    void Add(std::string const& URI, const std::shared_ptr<DataEntity>&) override;
    size_type Delete(std::string const& URI) override;
    size_type size() const override;
    size_type Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const override;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;

};  // class DataBackendXDMF {
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATABACKENDXDMF_H

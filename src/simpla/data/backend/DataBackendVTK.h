//
// Created by salmon on 17-8-14.
//

#ifndef SIMPLA_DATABACKENDVTK_H
#define SIMPLA_DATABACKENDVTK_H

#include "simpla/SIMPLA_config.h"

#include "../DataBackend.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataBackendVTK : public DataBackend {
    SP_OBJECT_HEAD(DataBackendVTK, DataBackend);

   public:
    DataBackendVTK();
    DataBackendVTK(DataBackendVTK const&);
    DataBackendVTK(DataBackendVTK&&) noexcept;
    explicit DataBackendVTK(std::string const& uri, std::string const& status = "");
    ~DataBackendVTK() override;

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
};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATABACKENDVTK_H

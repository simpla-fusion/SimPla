//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATABLE_H_
#define SIMPLA_DATATABLE_H_
#include "simpla/SIMPLA_config.h"

#include <memory>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTraits.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataBase;
/** @ingroup data */

class DataTable : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataTable, DataNode);
    struct pimpl_s;
    pimpl_s* m_pimpl_;

   protected:
    DataTable();

   public:
    ~DataTable() override;
    SP_DEFAULT_CONSTRUCT(DataTable);

    template <typename... Args>
    static std::shared_ptr<DataTable> New(Args&&... args) {
        return std::shared_ptr<DataTable>(new DataTable(std::forward<Args>(args)...));
    }
    size_type GetNumberOfChildren() const override;
    std::shared_ptr<DataNode> Child() const override;

    std::shared_ptr<DataNode> GetNodeByName(std::string const& s) const override;
    int SetNodeByName(std::string const& k, std::shared_ptr<DataNode> v) override;
    std::shared_ptr<DataNode> GetNodeByIndex(index_type s) const override { return GetNodeByName(std::to_string(s)); }
    int SetNodeByIndex(index_type idx, std::shared_ptr<DataNode> const& v) override {
        return SetNodeByName(std::to_string(idx), v);
    }
    int AddValue(std::shared_ptr<DataEntity> const& v) override {
        return SetNodeByIndex(GetNumberOfChildren(), DataNodeWithKey::New(v));
    }
    //******************************************************************************************************************
    /** Interface DataEntity */

    bool has(std::string const& uri) const { return Get(uri) != nullptr; }

    bool isNull() const;
    size_type Count() const;

    std::shared_ptr<DataEntity>& Get(std::string const& uri);
    std::shared_ptr<DataEntity> const& Get(std::string const& uri) const;
    int Set(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    int Add(std::string const& uri, const std::shared_ptr<DataEntity>& v);
    int Delete(std::string const& uri);
    int Set(const std::shared_ptr<DataTable>& v);
};

}  // namespace data

// template <typename U, typename... Args>
// std::shared_ptr<U> CreateObject(data::DataEntity const* dataEntity, Args&&... args) {
//    std::shared_ptr<U> res = nullptr;
//    if (dynamic_cast<data::DataLight<std::string> const*>(dataEntity) != nullptr) {
//        res = U::Create(dynamic_cast<data::DataLight<std::string> const*>(dataEntity)->value(),
//                        std::forward<Args>(args)...);
//    } else if (dynamic_cast<data::DataTable const*>(dataEntity) != nullptr) {
//        auto const* db = dynamic_cast<data::DataTable const*>(dataEntity);
//        res = U::Create(db->GetValue<std::string>("Type", ""), std::forward<Args>(args)...);
//        res->Deserialize(*db);
//    } else {
//        res = U::Create("", std::forward<Args>(args)...);
//    }
//    return res;
//};

}  // namespace simpla

#endif  // SIMPLA_DATATABLE_H_

//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include <string>
#include "../DataNode.h"

namespace simpla {
namespace data {

class DataNodeLua : public DataNode {
SP_DATA_NODE_HEAD(DataNodeLua)

   public:
    int Parse(std::string const& s) override;
    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    bool isValid() const override;

    size_type Set(index_type s, std::shared_ptr<DataEntity> const& v) override;
    size_type Add(index_type s, std::shared_ptr<DataEntity> const& v) override;
    size_type Delete(index_type s) override;
    std::shared_ptr<const DataNode> Get(index_type s) const override;
};
}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H

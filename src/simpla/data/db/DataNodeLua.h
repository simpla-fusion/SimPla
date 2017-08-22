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

class DataEntity;

class DataNodeLua : public DataNode {
    SP_DATA_NODE_HEAD(DataNodeLua)
};
}  // { namespace data {
}  // namespace simpla
#endif  // SIMPLA_LUADATABASE_H

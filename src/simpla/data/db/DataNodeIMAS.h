//
// Created by salmon on 17-8-24.
//

#ifndef SIMPLA_DATANODEIMAS_H
#define SIMPLA_DATANODEIMAS_H
#include <string>
#include "simpla/data/DataNode.h"
#include "simpla/utilities/ObjectHead.h"
namespace simpla {
namespace data {

class DataNodeIMAS : public DataNode {
SP_DATA_NODE_HEAD(DataNodeIMAS)
};
}  // { namespace data {
}  // namespace simpla

#endif  // SIMPLA_DATANODEIMAS_H

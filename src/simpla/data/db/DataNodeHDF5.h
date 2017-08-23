//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATABACKENDHDF5_H
#define SIMPLA_DATABACKENDHDF5_H

#include <string>
#include "simpla/data/DataNode.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
struct DataNodeHDF5 : public DataNode {
    SP_DATA_NODE_HEAD(DataNodeHDF5);
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

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

   public:
    int Connect(std::string const &authority, std::string const &path, std::string const &query,
                std::string const &fragment) override;

    int Disconnect() override;

    bool isValid() const override;

    int Flush() override;

    void Clear() override;
};
}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

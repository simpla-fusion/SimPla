//
// Created by salmon on 17-8-13.
//

#ifndef SIMPLA_DATABACKENDXDMF_H
#define SIMPLA_DATABACKENDXDMF_H

#include "../DataNode.h"
namespace simpla {
namespace data {
class DataNodeXDMF : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeXDMF, DataNode);
    SP_DATA_NODE_HEAD(DataNodeXDMF);

    SP_DATA_NODE_FUNCTION;

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};  // class DataNodeXDMF {
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATABACKENDXDMF_H

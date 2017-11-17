//
// Created by salmon on 17-9-1.
//

#ifndef SIMPLA_DATANODEMEMORY_H
#define SIMPLA_DATANODEMEMORY_H

#include <simpla/utilities/Factory.h>
#include "../DataBlock.h"
#include "../DataEntry.h"
#include "../Serializable.h"
namespace simpla {
namespace data {

struct DataEntryMemory : public DataEntry {
    SP_CREATABLE_HEAD(DataEntry, DataEntryMemory, mem)
    SP_DATA_NODE_FUNCTION(DataEntryMemory)

   private:
    std::map<std::string, std::shared_ptr<DataEntry>> m_table_;
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATANODEMEMORY_H

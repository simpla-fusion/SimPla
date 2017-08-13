//
// Created by salmon on 16-12-31.
//

#ifndef SIMPLA_DATA_ALL_H
#define SIMPLA_DATA_ALL_H

#include "Properties.h"
#include "DataArray.h"
#include "DataBackend.h"
#include "DataBlock.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "DataTraits.h"
#include "DataUtility.h"
#include "simpla/engine/EnableCreateFromDataTable.h"
#include "Serializable.h"

namespace simpla {
namespace data {
/**
 *  @addtogroup data Data
 *  @brief Unified data/metadata model.
 *  @details
 *  ## Summary
 *  ### Internal interface
 *   - @ref DataEntity
 *   - @ref DataLight
 *   - @ref DataHeavy
 *   - @ref DataType
 *   - @ref DataSpace
 *   - @ref DataSet
 *  ### External interface
 *   - @ref MPIDataType
 *   - @ref HDF5DataTable
 *   - @ref LuaDataTable
 *
 */
}
}

#endif  // SIMPLA_DATA_ALL_H

//
// Created by salmon on 16-12-31.
//

#ifndef SIMPLA_DATA_ALL_H
#define SIMPLA_DATA_ALL_H

#include "DataEntity.h"
#include "LightData.h"
#include "HeavyData.h"
#include "DataTable.h"


namespace simpla { namespace data
{
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
}}

#endif //SIMPLA_DATA_ALL_H

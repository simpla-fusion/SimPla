//
// Created by salmon on 16-12-31.
//

#ifndef SIMPLA_DATA_ALL_H
#define SIMPLA_DATA_ALL_H

#include "../../../experiment/DataBase.h"
#include "DataBlock.h"
#include "DataEntity.h"
#include "DataLight.h"
#include "DataNode.h"
#include "DataTraits.h"
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
    * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group
    * in HDF5,  all node/table are DataEntity.
    * @design_pattern
    *  - Proxy for DataBackend
    *
    *  PUT and POST are both unsafe methods. However, PUT is idempotent, while POST is not.
    *
    *  HTTP/1.1 SPEC
    *  @quota
    *   The POST method is used to request that the origin server accept the entity enclosed in
    *   the request as a new subordinate of the resource identified by the Request-URI in the Request-Line
    *
    *  @quota
    *  The PUT method requests that the enclosed entity be stored under the supplied Request-URI.
    *  If the Request-URI refers to an already existing resource, the enclosed entity SHOULD be considered as a
    *  modified version of the one residing on the origin server. If the Request-URI does not point to an existing
    *  resource, and that URI is capable of being defined as a new resource by the requesting user agent, the origin
    *  server can create the resource with that URI."
    *
    */
}
}

#endif  // SIMPLA_DATA_ALL_H

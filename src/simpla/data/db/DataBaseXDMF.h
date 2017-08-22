//
// Created by salmon on 17-8-13.
//

#ifndef SIMPLA_DATABACKENDXDMF_H
#define SIMPLA_DATABACKENDXDMF_H

#include "../../../../experiment/DataBase.h"
namespace simpla {
namespace data {

class DataBaseXDMF : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseXDMF);
};  // class DataBaseXDMF {
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATABACKENDXDMF_H

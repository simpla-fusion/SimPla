//
// Created by salmon on 17-8-14.
//

#ifndef SIMPLA_DATABACKENDVTK_H
#define SIMPLA_DATABACKENDVTK_H

#include "simpla/SIMPLA_config.h"
#include "../../../../experiment/DataBase.h"
namespace simpla {
namespace data {

class DataBaseVTK : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseVTK);
};

}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATABACKENDVTK_H

//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATABACKENDHDF5_H
#define SIMPLA_DATABACKENDHDF5_H

#include "simpla/data/DataBase.h"

#include <string>
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {
class DataBaseHDF5 : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseHDF5)
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDHDF5_H

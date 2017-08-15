//
// Created by salmon on 17-8-15.
//

#ifndef SIMPLA_DATABACKENDMDS_H
#define SIMPLA_DATABACKENDMDS_H
#include "simpla/data/DataBase.h"

namespace simpla {
namespace data {
class DataBaseMDS : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseMDS);
   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;
};

}  // namespace data{
}  // namespace simpla{

#endif  // SIMPLA_DATABACKENDMDS_H

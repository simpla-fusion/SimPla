//
// Created by salmon on 17-8-13.
//

#ifndef SIMPLA_DATABACKENDXDMF_H
#define SIMPLA_DATABACKENDXDMF_H

#include "../DataBase.h"
namespace simpla {
namespace data {

class DataBaseXDMF : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseXDMF);

   private:
    struct pimpl_s;
    pimpl_s* m_pimpl_ = nullptr;

};  // class DataBaseXDMF {
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATABACKENDXDMF_H

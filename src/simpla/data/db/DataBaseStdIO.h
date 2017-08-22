//
// Created by salmon on 17-8-16.
//

#ifndef SIMPLA_DATABASESTDIO_H
#define SIMPLA_DATABASESTDIO_H

#include <ostream>
#include <typeindex>

#include "../../../../experiment/DataBase.h"

namespace simpla {
namespace data {
class DataBaseStdIO : public DataBase {
    SP_DATABASE_DECLARE_MEMBERS(DataBaseStdIO)

    void SetStream(std::ostream&);
    void SetStream(std::istream&);
};  // class DataBase {
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABASESTDIO_H

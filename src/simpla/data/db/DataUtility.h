//
// Created by salmon on 17-3-9.
//

#ifndef SIMPLA_DATAUTILITY_H
#define SIMPLA_DATAUTILITY_H

#include <iomanip>
#include <regex>
#include <string>
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_traits.h"
namespace simpla {
namespace data {
class DataTable;
class DataEntity;
std::shared_ptr<DataTable> ParseCommandLine(int argc, char **argv);

template <typename U>
std::shared_ptr<DataTable> const &Pack(U const &u, ENABLE_IF((std::is_base_of<DataEntity, U>::value))) {
    return u.Pack();
}
template <typename U>
std::shared_ptr<U> Unpack(std::shared_ptr<DataTable> const &d, ENABLE_IF((std::is_base_of<Factory<U>, U>::value))) {
    return U::Create(d);
}
void Pack(std::shared_ptr<DataEntity> const &d, std::ostream &os, std::string const &type, int indent = 0);
std::string AutoIncreaseFileName(std::string filename, std::string const &ext_str);

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAUTILITY_H

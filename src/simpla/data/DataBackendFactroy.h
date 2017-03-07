//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATATABLEFACTROY_H
#define SIMPLA_DATATABLEFACTROY_H

#include <memory>
#include <string>
namespace simpla {
namespace data {
class DataBackend;
DataBackend* CreateDataBackendFromFile(std::string const& url);
}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATATABLEFACTROY_H

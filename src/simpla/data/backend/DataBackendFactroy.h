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
std::unique_ptr<DataBackend> CreateDataBackend(std::string const &url = "", std::string const &param = "");
}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATATABLEFACTROY_H

//
// Created by salmon on 17-8-24.
//

#include "DataNodeIMAS.h"
namespace simpla {
namespace data {

REGISTER_CREATOR(DataNodeIMAS, imas);
DataNodeIMAS::DataNodeIMAS() {}
DataNodeIMAS::~DataNodeIMAS() {}

int DataNodeIMAS::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    return 0;
}
int DataNodeIMAS::Disconnect() { return 0; }
bool DataNodeIMAS::isValid() const { return false; }

}  // { namespace data {
}  // namespace simpla
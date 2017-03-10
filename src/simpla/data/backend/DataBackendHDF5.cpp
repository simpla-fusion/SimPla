//
// Created by salmon on 17-3-10.
//

#include "DataBackendHDF5.h"
#include <simpla/design_pattern/SingletonHolder.h>
#include "../DataBackendFactory.h"
namespace simpla {
namespace data {
REGISTER_DATA_BACKEND_CREATOR(DataBackendHDF5, h5)
constexpr char DataBackendHDF5::ext[];

struct DataBackendHDF5::pimpl_s {};
DataBackendHDF5::DataBackendHDF5() : m_pimpl_(new pimpl_s) {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5 const& other) : DataBackendHDF5() {} /* copy pimpl_s*/
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendHDF5::DataBackendHDF5(std::string const& uri, std::string const& status) : DataBackendHDF5() {
    UNIMPLEMENTED;
}
DataBackendHDF5::~DataBackendHDF5() {}
std::ostream& DataBackendHDF5::Print(std::ostream& os, int indent) const { return os; }

std::unique_ptr<DataBackend> DataBackendHDF5::CreateNew() const { return std::make_unique<DataBackendHDF5>(); }
void DataBackendHDF5::Flush() { UNIMPLEMENTED; }
bool DataBackendHDF5::isNull() const { UNIMPLEMENTED; }
size_type DataBackendHDF5::size() const { UNIMPLEMENTED; }

std::shared_ptr<DataEntity> DataBackendHDF5::Get(std::string const& URI) const { UNIMPLEMENTED; }
std::shared_ptr<DataEntity> DataBackendHDF5::Get(id_type key) const { UNIMPLEMENTED; }
bool DataBackendHDF5::Set(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendHDF5::Set(id_type key, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendHDF5::Add(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendHDF5::Add(id_type key, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
size_type DataBackendHDF5::Delete(std::string const& URI) { UNIMPLEMENTED; }
size_type DataBackendHDF5::Delete(id_type key) { UNIMPLEMENTED; }
void DataBackendHDF5::DeleteAll() { UNIMPLEMENTED; }

size_type DataBackendHDF5::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}
size_type DataBackendHDF5::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}

}  // namespace data{
}  // namespace simpla{
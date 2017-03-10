//
// Created by salmon on 17-3-10.
//

#include "DataBackendSAMRAI.h"

namespace simpla {
namespace data {
constexpr char DataBackendSAMRAI::ext[]  ;

struct DataBackendSAMRAI::pimpl_s {};
DataBackendSAMRAI::DataBackendSAMRAI() : m_pimpl_(new pimpl_s) {}
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI const& other) : DataBackendSAMRAI() {} /* copy pimpl_s*/
DataBackendSAMRAI::DataBackendSAMRAI(DataBackendSAMRAI&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendSAMRAI::DataBackendSAMRAI(std::string const& uri, std::string const& status) : DataBackendSAMRAI() {
    UNIMPLEMENTED;
}
DataBackendSAMRAI::~DataBackendSAMRAI() {}
std::ostream& DataBackendSAMRAI::Print(std::ostream& os, int indent) const { return os; }

std::unique_ptr<DataBackend> DataBackendSAMRAI::CreateNew() const { return std::make_unique<DataBackendSAMRAI>(); }
void DataBackendSAMRAI::Flush() { UNIMPLEMENTED; }
bool DataBackendSAMRAI::isNull() const { UNIMPLEMENTED; }
size_type DataBackendSAMRAI::size() const { UNIMPLEMENTED; }

std::shared_ptr<DataEntity> DataBackendSAMRAI::Get(std::string const& URI) const { UNIMPLEMENTED; }
std::shared_ptr<DataEntity> DataBackendSAMRAI::Get(id_type key) const { UNIMPLEMENTED; }
bool DataBackendSAMRAI::Set(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendSAMRAI::Set(id_type key, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendSAMRAI::Add(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
bool DataBackendSAMRAI::Add(id_type key, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
size_type DataBackendSAMRAI::Delete(std::string const& URI) { UNIMPLEMENTED; }
size_type DataBackendSAMRAI::Delete(id_type key) { UNIMPLEMENTED; }
void DataBackendSAMRAI::DeleteAll() { UNIMPLEMENTED; }

size_type DataBackendSAMRAI::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}
size_type DataBackendSAMRAI::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}

}  // namespace data{
}  // namespace simpla{
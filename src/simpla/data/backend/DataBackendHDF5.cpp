//
// Created by salmon on 17-3-10.
//

#include "DataBackendHDF5.h"
#include <simpla/algebra/Array.h>
#include <simpla/algebra/nTuple.h>

#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
extern "C" {
#include <hdf5.h>
#include <hdf5_hl.h>
}

#include <H5FDmpio.h>

namespace simpla {
namespace data {
constexpr char DataBackendHDF5::scheme_tag[];
std::string DataBackendHDF5::scheme() const { return scheme_tag; }

// struct HDF5Status {
//    HDF5Status(std::string const& url, std::string const& status = "");
//    ~HDF5Status();
//    hid_t base_file_id_ = -1;
//    hid_t base_group_id_ = -1;
//    void Close();
//    void Open(std::string const& url, std::string const& status);
//};
// HDF5Status::HDF5Status(std::string const& url, std::string const& status) { Open(url, status); }
// HDF5Status::~HDF5Status() {
//    if (base_file_id_ != -1) { Close(); }
//}
// class HDF5Closer {
//    void* addr_;
//    size_t s_;
//    HDF5Closer(void* p, size_t s) : addr_(p), s_(s) {}
//    ~HDF5Closer() {}
//    inline void operator()(void* ptr) {
//        hid_t f = *reinterpret_cast<hid_t*>(ptr);
//        if (f != -1) { H5Fclose(f); }
//    }
//};

// template <typename U, int NDIMS>
// class DataEntityHeavyArrayHDF5 : public DataEntityWrapper<Array<U, NDIMS>> {};
struct DataBackendHDF5::pimpl_s {
    std::shared_ptr<hid_t> m_hdf5_;
};
DataBackendHDF5::DataBackendHDF5() : m_pimpl_(new pimpl_s) {}
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5 const& other) : DataBackendHDF5() {} /* copy pimpl_s*/
DataBackendHDF5::DataBackendHDF5(DataBackendHDF5&& other) : m_pimpl_(std::move(m_pimpl_)) {}
DataBackendHDF5::DataBackendHDF5(std::string const& uri, std::string const& status) : DataBackendHDF5() {}
DataBackendHDF5::~DataBackendHDF5() {}
std::ostream& DataBackendHDF5::Print(std::ostream& os, int indent) const { return os; }
std::shared_ptr<DataBackend> DataBackendHDF5::Clone() const { return std::make_shared<DataBackendHDF5>(*this); }
std::shared_ptr<DataBackend> DataBackendHDF5::Create() const { return std::make_shared<DataBackendHDF5>(); }

void DataBackendHDF5::Flush() { UNIMPLEMENTED; }
bool DataBackendHDF5::isNull() const { UNIMPLEMENTED; }
size_type DataBackendHDF5::size() const { UNIMPLEMENTED; }

std::shared_ptr<DataEntity> DataBackendHDF5::Get(std::string const& URI) const { UNIMPLEMENTED; }
void DataBackendHDF5::Set(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
void DataBackendHDF5::Add(std::string const& URI, std::shared_ptr<DataEntity> const&) { UNIMPLEMENTED; }
size_type DataBackendHDF5::Delete(std::string const& URI) { UNIMPLEMENTED; }

size_type DataBackendHDF5::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const {
    UNIMPLEMENTED;
}

}  // namespace data{
}  // namespace simpla{
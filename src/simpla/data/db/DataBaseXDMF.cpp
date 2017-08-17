//
// Created by salmon on 17-8-13.
//
#include "DataBaseXDMF.h"
#include <simpla/data/DataTable.h>
#include <sys/stat.h>
#include <Xdmf.hpp>
#include <XdmfDomain.hpp>
#include <XdmfWriter.hpp>

namespace simpla {
namespace data {

struct DataBackendXDMFRoot : public DataBaseXDMF {
    boost::shared_ptr<XdmfDomain> m_root_ = nullptr;
    void Initialize();
    int Flush();
//    int Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity);

   private:
    std::string m_prefix_ = "simpla_unamed";
    std::string m_suffix_ = "xmf";
    std::string m_pwd_;
    size_type m_counter_ = 0;
    size_type m_local_rank_ = 0;
    std::string m_local_prefix_;
    Real m_time_ = 0;
};

// int DataBackendXDMFRoot::Connect(std::string const& authority, std::string const& path, std::string const& query,
//                                 std::string const& fragment) {
//    m_prefix_ = path;
//}
#define NUMBER_OF_FILE_COUNT_DIGTALS 6

std::string add_count_suffix(std::string const& prefix, size_type n, int width = NUMBER_OF_FILE_COUNT_DIGTALS) {
    std::ostringstream os;
    os << prefix.c_str() << "." << std::setw(width) << std::setfill('0') << n;
    return os.str();
}
void DataBackendXDMFRoot::Initialize() {
    if (!m_pwd_.empty()) { return; }
    Flush();
    m_pwd_ = add_count_suffix(m_prefix_, m_counter_);
    m_local_prefix_ = add_count_suffix(m_prefix_ + "/summary", m_local_rank_);
    auto error_code = mkdir(m_pwd_.c_str(), 0777);
    if (error_code != SP_SUCCESS) { RUNTIME_ERROR << "Can not make directory [" << m_pwd_ << "]" << std::endl; }

    m_root_ = XdmfDomain::New();
}
int DataBackendXDMFRoot::Flush() {
    if (m_root_ != nullptr) { m_root_->accept(XdmfWriter::New(m_local_prefix_ + ".xmf")); }
    m_pwd_ = "";
    ++m_counter_;
    return 0;
}
struct DataBaseXDMF::pimpl_s {
    std::string m_name_;
    std::shared_ptr<DataBaseXDMF> m_parent_ = nullptr;
    std::shared_ptr<XdmfItem> m_self_;
};
DataBaseXDMF::DataBaseXDMF() : m_pimpl_(new pimpl_s) {}
DataBaseXDMF::~DataBaseXDMF() {
    Flush();
    delete m_pimpl_;
}

bool DataBaseXDMF::isNull() const { return m_pimpl_->m_self_ == nullptr; }

int DataBaseXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                          std::string const& fragment) {
    m_pimpl_->m_parent_->Connect(authority, path, query, fragment);
    return SP_SUCCESS;
}

int DataBaseXDMF::Disconnect() { return SP_SUCCESS; }

int DataBaseXDMF::Flush() { return 0; }

std::shared_ptr<DataEntity> DataBaseXDMF::Get(std::string const& URI) const {
    if (URI[0] == '/') {
        //        GetRoot()->Get(URI.substr(1));
    } else {
        auto new_backend = DataBaseXDMF::New();
        new_backend->m_pimpl_->m_parent_ =
            std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());
        new_backend->m_pimpl_->m_name_ = URI;
        auto res = DataTable::New(new_backend);
    }
    return nullptr;
}
int DataBaseXDMF::Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity) { return 0; }
int DataBaseXDMF::Add(std::string const& URI, const std::shared_ptr<DataEntity>& entity) { return 0; }
int DataBaseXDMF::Delete(std::string const& URI) { return 0; }
int DataBaseXDMF::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
    return 0;
}

}  // namespace data{
}  // namespace simpla
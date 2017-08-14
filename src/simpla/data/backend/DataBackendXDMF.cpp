//
// Created by salmon on 17-8-13.
//
#include "DataBackendXDMF.h"
#include <sys/stat.h>
#include <Xdmf.hpp>
#include <XdmfDomain.hpp>
#include <XdmfWriter.hpp>

namespace simpla {
namespace data {

struct DataBackendXDMFRoot : public DataBackendXDMF {
    boost::shared_ptr<XdmfDomain> m_root_ = nullptr;
    void Initialize();
    void Flush();
    void Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity) {}
    DataBackendXDMF* GetRoot() override;
    DataBackendXDMF const* GetRoot() const override;

   private:
    std::string m_prefix_ = "simpla_unamed";
    std::string m_suffix_ = "xmf";
    std::string m_pwd_;
    size_type m_counter_ = 0;
    size_type m_local_rank_ = 0;
    std::string m_local_prefix_;
    Real m_time_ = 0;
};

DataBackendXDMF* DataBackendXDMFRoot::GetRoot() { return this; }
DataBackendXDMF const* DataBackendXDMFRoot::GetRoot() const { return this; }

void DataBackendXDMFRoot::Connect(std::string const& authority, std::string const& path, std::string const& query,
                                  std::string const& fragment) {
    m_prefix_ = path;
}
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
void DataBackendXDMFRoot::Flush() {
    if (m_root_ != nullptr) { m_root_->accept(XdmfWriter::New(m_local_prefix_ + ".xmf")); }
    m_pwd_ = "";
    ++m_counter_;
}
struct DataBackendXDMF::pimpl_s {
    std::string m_name_;
    std::shared_ptr<DataBackendXDMF> m_parent_ = nullptr;
    std::shared_ptr<XdmfItem> m_self_;
};
DataBackendXDMF::DataBackendXDMF() : m_pimpl_(new pimpl_s) {}
DataBackendXDMF::~DataBackendXDMF() {
    Flush();
    delete m_pimpl_;
}
DataBackendXDMF* DataBackendXDMF::GetRoot() {
    return m_pimpl_->m_parent_ == nullptr ? nullptr : m_pimpl_->m_parent_->GetRoot();
};
DataBackendXDMF const* DataBackendXDMF::GetRoot() const {
    return m_pimpl_->m_parent_ == nullptr ? nullptr : m_pimpl_->m_parent_->GetRoot();
};

DataBackendXDMF::DataBackendXDMF(std::string const& uri, std::string const& status) {}

bool DataBackendXDMF::isNull() const { return m_pimpl_->m_self_ == nullptr; }

void DataBackendXDMF::Connect(std::string const& authority, std::string const& path, std::string const& query,
                              std::string const& fragment) {
    m_pimpl_->m_parent_->Connect(authority, path, query, fragment);
}

void DataBackendXDMF::Disconnect() {}

std::shared_ptr<DataBackend> DataBackendXDMF::Duplicate() const {
    auto res = std::dynamic_pointer_cast<DataBackendXDMF>(CreateNew());

    return res;
}
std::shared_ptr<DataBackend> DataBackendXDMF::CreateNew() const {
    auto res = std::make_shared<DataBackendXDMF>();
    res->m_pimpl_->m_parent_ = m_pimpl_->m_parent_;
}
void DataBackendXDMF::Flush() { GetRoot()->Flush(); }

std::shared_ptr<DataEntity> DataBackendXDMF::Get(std::string const& URI) const {
    if (URI[0] == '/') {
        GetRoot()->Get(URI.substr(1));
    } else {
        auto new_backend = std::make_shared<DataBackendXDMF>();
        new_backend->m_pimpl_->m_parent_ = this->shared_from_this();
        new_backend->m_pimpl_->m_name_ = URI;
        auto res = std::make_shared<DataTable>(new_backend);
    }
    return nullptr;
}
void DataBackendXDMF::Set(std::string const& URI, const std::shared_ptr<DataEntity>& entity) {}
void DataBackendXDMF::Add(std::string const& URI, const std::shared_ptr<DataEntity>& entity) {}
size_type DataBackendXDMF::Delete(std::string const& URI) { return 0; }
size_type DataBackendXDMF::size() const { return 0; }
size_type DataBackendXDMF::Foreach(
    std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& fun) const {
    return 0;
}

}  // namespace data{
}  // namespace simpla
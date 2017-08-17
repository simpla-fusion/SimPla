//
// Created by salmon on 17-8-16.
//
#include "DataBaseStdIO.h"
#include "../DataEntityVisitor.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseStdIO, stdio);

struct DataBaseStdIO::pimpl_s {
    std::ostream* m_out_ = nullptr;
    std::istream* m_in_ = nullptr;
};
DataBaseStdIO::DataBaseStdIO() : m_pimpl_(new pimpl_s) {}
DataBaseStdIO::~DataBaseStdIO() { delete m_pimpl_; }
void DataBaseStdIO::SetStream(std::ostream& out) { m_pimpl_->m_out_ = &out; }
void DataBaseStdIO::SetStream(std::istream& in) { m_pimpl_->m_in_ = &in; }
int DataBaseStdIO::Connect(std::string const& authority, std::string const& path, std::string const& query,
                           std::string const& fragment) {
    m_pimpl_->m_out_ = &(std::cout);
    m_pimpl_->m_in_ = &(std::cin);
    return SP_SUCCESS;
}

int DataBaseStdIO::Disconnect() {
    m_pimpl_->m_out_ = nullptr;
    m_pimpl_->m_in_ = nullptr;
    return SP_SUCCESS;
}
int DataBaseStdIO::Flush() {
    *m_pimpl_->m_out_ << std::endl;
    return SP_SUCCESS;
}
bool DataBaseStdIO::isNull(std::string const& uri) const { return m_pimpl_->m_out_ == nullptr; }
size_type DataBaseStdIO::Count(std::string const& url) const { return 0; }
std::shared_ptr<DataEntity> DataBaseStdIO::Get(std::string const& URI) const { return nullptr; }

struct VisitorStdOut : public DataEntityVisitor {
    std::ostream* m_out_;

    int visit(int u) override { return Print(u); }
    int visit(Real u) override { return Print(u); }
    int visit(std::complex<Real> const& u) override { return Print(u); }
    int visit(std::string const& u) override { return Print(u); }
    int visit(int const* u, int ndims, int const* d) override { return 0; }
    int visit(Real const* u, int ndims, int const* d) override { return 0; }
    int visit(std::complex<Real> const* u, int ndims, int const* d) override { return 0; }
    int visit(std::string const* u, int ndims, int const* d) override { return 0; }

    template <typename T>
    int Print(T const& v) {
        *m_out_ << v;
        return 1;
    }
};
std::ostream& Print(std::ostream& os, std::shared_ptr<DataEntity> const& v, int indent){

};

int DataBaseStdIO::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    *m_pimpl_->m_out_ << uri << "=";
    Print(*m_pimpl_->m_out_, v, 0);
    return 0;
}
int DataBaseStdIO::Add(std::string const& URI, const std::shared_ptr<DataEntity>& v) { return 0; }
int DataBaseStdIO::Delete(std::string const& URI) { return 0; }
int DataBaseStdIO::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return 0;
}
}
}
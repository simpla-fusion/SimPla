//
// Created by salmon on 17-3-10.
//

#ifndef SIMPLA_DATABACKENDSAMRAI_H
#define SIMPLA_DATABACKENDSAMRAI_H
#include "../DataBackend.h"
namespace simpla {
namespace data {
class DataBackendSAMRAI : public DataBackend {
   public:
    static constexpr char scheme_tag[] = "samrai";

    DataBackendSAMRAI();
    DataBackendSAMRAI(DataBackendSAMRAI const&);
    DataBackendSAMRAI(DataBackendSAMRAI&&);

    DataBackendSAMRAI(std::string const& uri, std::string const& status = "");
    virtual ~DataBackendSAMRAI();

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const;
    virtual std::string scheme() const;
    virtual std::shared_ptr<DataBackend> Clone() const;
    virtual std::shared_ptr<DataBackend> Create() const;
    virtual bool isNull() const;
    virtual void Flush();

    virtual std::shared_ptr<DataEntity> Get(std::string const& URI) const;
     virtual void Set(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual void Add(std::string const& URI, std::shared_ptr<DataEntity> const&);
    virtual size_type Delete(std::string const& URI);
    virtual size_type size() const;
    virtual size_type Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const&) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATABACKENDSAMRAI_H

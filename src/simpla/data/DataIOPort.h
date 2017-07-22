//
// Created by salmon on 17-7-19.
//

#ifndef SIMPLA_DATAIOPORT_H
#define SIMPLA_DATAIOPORT_H

#include "DataArray.h"
#include "EnableCreateFromDataTable.h"
#include "simpla/utilities/ObjectHead.h"

namespace simpla {
namespace data {

class DataIOPort : public EnableCreateFromDataTable<DataIOPort> {
    SP_OBJECT_HEAD(DataIOPort, EnableCreateFromDataTable<DataIOPort>);

   public:
    explicit DataIOPort(std::string uri = "");
    ~DataIOPort() override;

    SP_DEFAULT_CONSTRUCT(DataIOPort)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    id_type TryGet(std::string const &uri, std::shared_ptr<DataEntity> *) const;
    id_type TrySet(std::string const &uri, std::shared_ptr<DataEntity> const &);
    id_type TryAdd(std::string const &uri, std::shared_ptr<DataEntity> const &);
    id_type TryDelete(std::string const &uri);
    id_type Cancel(id_type);
    bool Check(id_type) const;
    void Flush();

    std::shared_ptr<DataEntity> Get(std::string const &uri) const;
    void Set(std::string const &uri, std::shared_ptr<DataEntity> const &);
    void Add(std::string const &uri, std::shared_ptr<DataEntity> const &);

    void Set(std::shared_ptr<DataTable> const &);
    void Add(std::shared_ptr<DataTable> const &);

    int Delete(std::string const &uri);

    template <typename... Args>
    void Set(std::string const &uri, Args &&... args) {
        Set(uri, make_data_entity(std::forward<Args>(args)...));
    }

    template <typename... Args>
    void Add(std::string const &uri, Args &&... args) {
        Add(uri, make_data_entity(std::forward<Args>(args)...));
    }

    template <typename T, typename... Args>
    T Get(Args &&... args) const {
        return Get(std::forward<Args>(args)...)->template cast_as<T>();
    }

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAIOPORT_H

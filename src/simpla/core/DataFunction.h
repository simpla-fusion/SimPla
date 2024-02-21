//
// Created by salmon on 17-8-25.
//

#ifndef SIMPLA_DATAFUNCTION_H
#define SIMPLA_DATAFUNCTION_H

#include "DataEntity.h"
#include "DataUtilities.h"
namespace simpla {
namespace data {
struct DataFunction : public DataEntity {
    SP_DEFINE_FANCY_TYPE_NAME(DataFunction, DataEntity)
   protected:
    DataFunction() = default;

   public:
    ~DataFunction() = default;
    template <typename... Args>
    static std::shared_ptr<DataFunction> New(Args&&... args) {
        return std::shared_ptr<DataFunction>(new DataFunction(std::forward<Args>(args)...));
    }

    std::ostream& Print(std::ostream& os, int indent) const override {
        os << "<Function>";
        return os;
    }

    virtual std::shared_ptr<DataEntity> eval(std::initializer_list<std::shared_ptr<DataEntity>> const& args) const {
        return DataEntity::New();
    };
    template <typename U, typename... Args>
    U as(Args&&... args) {
        return data_cast<U>(eval({make_data(std::forward<Args>(args))...}));
    };
    template <typename... Args>
    std::shared_ptr<DataEntity> operator()(Args&&... args) const {
        return eval({make_data(std::forward<Args>(args))...});
    }
};
}  // namespace data
}  // namespace simpla
#endif  // SIMPLA_DATAFUNCTION_H

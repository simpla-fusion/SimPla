//
// Created by salmon on 17-5-1.
//

#ifndef SIMPLA_MODELLOADER_H
#define SIMPLA_MODELLOADER_H

#include <simpla/engine/Model.h>
namespace simpla {
namespace model {

class ModelLoader : EnableCreateFromDataTable<ModelLoader> {
   public:
    ModelLoader();
    ~ModelLoader() override;
    SP_DEFAULT_CONSTRUCT(ModelLoader)
    DECLARE_REGISTER_NAME("ModelLoader")
    void Load(Model *);
};
}  // namespace model {
}  // namespace simpla {

#endif  // SIMPLA_MODELLOADER_H

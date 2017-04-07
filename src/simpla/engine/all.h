//
// Created by salmon on 17-3-19.
//

#ifndef SIMPLA_ENGINE_ALL_H
#define SIMPLA_ENGINE_ALL_H

#include "Atlas.h"
#include "Attribute.h"
#include "Context.h"
#include "Domain.h"

#include "Mesh.h"
#include "Model.h"
#include "Patch.h"
#include "Task.h"
#include "TimeIntegrator.h"
/**
 * @startuml
 * class Worker{
 * virtual void Register(AttributeBundle*)
 * }
 * class Field<TMesh>{
 * }
 * class TaskTemplate<TMesh>{
 * }
 * class Task{
 *      virtual void Register(AttributeBundle*)
 * }
 * class Mesh {
 *      virtual void Register(AttributeBundle*)
 * }
 * class Attribute {
 *      Attribute(AttributeBundle*);
 *      void Register(AttributeBundle*)
 *      void SetMesh(Mesh *);
 *      void SetRange(IdRange *);
 *      Push(DataBlock);
 *      DataBlock Pop();
 * }
 * class AttributeBundle {
 *      void Register(AttributeBundle*);
 *      void Detach(Attribute *attr);
 *      void Attach(Attribute *attr);
 *      virtual void Push(Patch );
 *      virtual Patch Pop();
 * }
 * class IndexSpace{
 *      point_type origin;
 *      point_type dx;
 * }
 * Context o-- "0..*" Domain
 * Context o-- "0..*" Mesh
 * Context o-- "0..*" GeoObject
 * Context *-- "1" Atlas
 * Context *-- "1" AttributeBundle
 * Atlas o-- "0..*" Patch
 * Domain *-- "1" GeoObject
 * Domain *-- "1" Mesh
 * Domain o-- "0..*" Worker
 * Domain *-- "1" AttributeBundle
 * Domain ..> IdRange: create
 * GeoObject  .. Mesh
 * IdRange . (GeoObject,Mesh)
 * IdRange --* Attribute
 * Worker *--  Mesh
 * Worker o--  Attribute
 * Attribute *-- Mesh
 * IndexSpace "1" o-- "0..*" MeshBlock
 * Patch *-- "1" MeshBlock
 * Patch *-- "0..*" DataBlock
 * DataBlock ..> MeshBlock
 * Mesh *-- "1" IndexSpace
 * Attribute <|-- Field
 *
 * TMesh --|> Mesh
 * Field .. TMesh
 * TaskTemplate --|> Task
 * TaskTemplate .. TMesh
 * TaskTemplate o-- Field
 * Worker o-- Task
 * Patch <..> AttributeBundle : push/pop
 * DataBlock <..>Attribute : push/pop
 * MeshBlock <..> Mesh: push/pop
 * @enduml
 * @startuml
 * :Domain1:
 * (Domain2)
 *
 *
 * @enduml
 *
 */

/**
 *
 * @startuml
 * actor Main
 * Main -> DomainView : Set U as MeshView
 * activate DomainView
 *     alt if MeshView=nullptr
 *          create MeshView
 *     DomainView -> MeshView : create U as MeshView
 *     MeshView --> DomainView: return MeshView
 *     end
 *     DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * actor Main
 * Main -> DomainView : Dispatch
 * activate DomainView
 *     DomainView->MeshView:  Dispatch
 *     MeshView->MeshView: SetMeshBlock
 *     activate MeshView
 *     deactivate MeshView
 *     MeshView -->DomainView:  Done
*      DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * Main ->DomainView: Update
 * activate DomainView
 *     DomainView -> AttributeView : Update
 *     activate AttributeView
 *          AttributeView -> Field : Update
 *          Field -> AttributeView : Update
 *          activate AttributeView
 *               AttributeView -> DomainView : get DataBlock at attr.id()
 *               DomainView --> AttributeView : return DataBlock at attr.id()
 *               AttributeView --> Field : return DataBlock is ready
 *          deactivate AttributeView
 *          alt if data_block.isNull()
 *              Field -> Field :  create DataBlock
 *              Field -> AttributeView : send DataBlock
 *              AttributeView --> Field : Done
 *          end
 *          Field --> AttributeView : Done
 *          AttributeView --> DomainView : Done
 *     deactivate AttributeView
 *     DomainView -> MeshView : Update
 *     activate MeshView
 *          alt if isFirstTime
 *              MeshView -> AttributeView : Set Initialize Value
 *              activate AttributeView
 *                   AttributeView --> MeshView : Done
 *              deactivate AttributeView
 *          end
 *          MeshView --> DomainView : Done
 *     deactivate MeshView
 *     DomainView -> Worker : Update
 *     activate Worker
 *          alt if isFirstTime
 *              Worker -> AttributeView : set initialize value
 *              activate AttributeView
 *                  AttributeView --> Worker : Done
 *              deactivate AttributeView
 *          end
 *          Worker --> DomainView : Done
 *     deactivate Worker
 *     DomainView --> Main : Done
 * deactivate DomainView
 * deactivate Main
 * @enduml
 */
#endif  // SIMPLA_ENGINE_ALL_H

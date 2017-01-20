//
// Created by salmon on 17-1-20.
//

#ifndef SIMPLA_ALL_H
#define SIMPLA_ALL_H
namespace simpla{
namespace mesh{
/**
 * @addtogroup mesh{
 *
 *
 * @startuml
 *
 *  Mesh "0" o-- "1" Model
 *  Mesh "1" o-- "1" MeshBlock
 *  Attribute "1" o-- "1" DataBlock
 *  Attribute "n" o-- "1" AttributeDesc
 *  Attribute <|-- AttributeAdapter
 *  Attribute <|-- DataAttribute
 *  Worker "1" *-- Model
 *  Worker "1" *-- Mesh
 *  Worker "1" o-- "0..n" Attribute
 * @enduml
 *
 * @}
 *
 */
}
}
#endif //SIMPLA_ALL_H

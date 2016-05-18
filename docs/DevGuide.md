Develop Guide
=============
[TOC]
    

## Name style and directory structure

* namespace : Namespace names are all lower-case. Top-level namespace names are based on the project name .
* type name : Type names start with a capital letter and have a capital letter for each new word, with no underscores: MyExcitingClass, MyExcitingEnum.
* file name : File are based on the class name.               
* module    :  
          - Each module have a namespace;  
          - Each module have a sub-directory, whose name is based on module name.
                Example:           
                <project root>/src/foo_bar/UBarBaz.h:
                namespace foo_bar
                {
                 class UBarBaz{
                  void balalal();
                 };
                }  
                <project root>/src/foo_bar/UBarBaz.cpp:
                void foo_bar::UBarBaz::balalal(){ /* do sth.*/}
* class concept:  Classes have same prefix, if they are in the same concept
                Example:
                class WriterHDF5;
                class WriterXDMF;
                class WriterVTK; 
                ....
              
  

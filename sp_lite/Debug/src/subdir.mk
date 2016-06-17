################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/spBucketFunction.cu \
../src/spField.cu \
../src/spMesh.cu \
../src/spParticle.cu 

C_SRCS += \
../src/spBucketFunction.c \
../src/spField.c \
../src/spMesh.c \
../src/spParticle.c 

OBJS += \
./src/spBucketFunction.o \
./src/spField.o \
./src/spMesh.o \
./src/spParticle.o 

CU_DEPS += \
./src/spBucketFunction.d \
./src/spField.d \
./src/spMesh.d \
./src/spParticle.d 

C_DEPS += \
./src/spBucketFunction.d \
./src/spField.d \
./src/spMesh.d \
./src/spParticle.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



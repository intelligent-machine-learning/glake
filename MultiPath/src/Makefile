CUDA_DIR=/usr/local/cuda

CUR_DIR=$(shell pwd)
GDR_INCLUDE=${CUR_DIR}/gdrcopy-2.4/include
GDR_LIB=${CUR_DIR}/gdrcopy-2.4/src

FLAGS = -I${CUDA_DIR}/include \
   -I${GDR_INCLUDE} \
   -L/usr/lib/x86_64-linux-gnu/ \
   -L${CUDA_DIR}/lib64 \
   -L${GDR_LIB} \
   -D_GNU_SOURCE \
   -g -O2 -DNDEBUG \
   -std=c++14 \
   -shared -lrt -ldl \
   -lpthread \
   -lgdrapi

NVFLAGS = \
   -Xcompiler "-fPIC" \
   -Xlinker "-rpath=${GDR_LIB}" \
   -Xlinker "-soname,libcuda.so.1" \
   -gencode arch=compute_70,code=sm_70 \
   -gencode arch=compute_75,code=sm_75 \
   -gencode arch=compute_80,code=sm_80 \
   -gencode arch=compute_86,code=sm_86

all: libcuda.so.1

libcuda.so.1:gmm_multipath_impl.cu gmm_common_impl.o gmm_client_cfg.o gmm_client_impl.o gmm_server_impl.o gmm_worker_impl.o gmm_host_shm_impl.o gmm_cuda_mem_impl.o gmm_gdr_plugin.o gmm_api_stats.o cuda.o
	mkdir -p ../bin
	nvcc -o ../bin/$@ $^ ${FLAGS} ${NVFLAGS}
	ln -sf ../bin/libcuda.so.1 ../bin/libcuda.so

%.o: %.cpp
	g++ -c -o $@ $^ -fPIC ${FLAGS}

format:
	clang-format-11 -i *.cu *.cpp *.h --style=google

clean:
	rm -f *.o ../bin/libcuda.so.1 ../bin/libcuda.so


all: gmm_test gmm_bench

gmm_test: gmm_test.cu
	nvcc -o $@ $^ -g -I../include -L../bin -lcuda

gmm_bench: gmm_bench.cu
	nvcc -o $@ $^ -g -I../include -L../bin -lcuda

bench: gmm_bench
	rm -f /tmp/gmm*
	rm -f /dev/shm/gmm_hShm*
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
		GMM_SYNC=0 \
		GMM_LOG=3 \
		GMM_MP=1 \
		GMM_GPU_BUF=128 \
		GMM_IPC_DIR=/tmp \
		GMM_REMOVE_SOCKET=1 \
		GMM_DP=1 \
		LD_LIBRARY_PATH=../bin/ \
		LD_PRELOAD=../bin/libcuda.so.1 \
		I=100 \
		N=512 \
		GLAKE_CACHE_MAX_SIZE=32 \
		GLAKE_CACHE_BLOCK_SIZE=8 \
		CUDA_BACK=0 \
		GLAKE_MULTI_PATH=1 \
	    ./gmm_bench

test: gmm_test
	rm -f /tmp/gmm*
	rm -f /dev/shm/gmm_hShm*
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
		GMM_SYNC=0 \
		GMM_LOG=3 \
		GMM_MP=1 \
		GMM_GPU_BUF=128 \
		GMM_IPC_DIR=/tmp \
		GMM_REMOVE_SOCKET=1 \
		GMM_DP=1 \
		LD_LIBRARY_PATH=../bin/ \
		LD_PRELOAD=../bin/libcuda.so.1 \
		I=1 \
		N=512 \
		GLAKE_CACHE_MAX_SIZE=32 \
		GLAKE_CACHE_BLOCK_SIZE=8 \
		CUDA_BACK=0 \
		GLAKE_MULTI_PATH=1 \
	    ./gmm_test

format:
	clang-format-11 -i *.cu *.h --style=google

clean:
	rm -f gmm_test gmm_bench

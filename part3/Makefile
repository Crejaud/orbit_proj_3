objects= bfs.o
NVCC= /usr/local/cuda-8.0/bin/nvcc
opt= -O2 -g -G
ARCH= -arch=sm_30
LIBS=
execname= bfs

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS)

bfs.o: bfs.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c bfs.cu

clean:
	rm $(objects)

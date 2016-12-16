objects= matr_mult.o
NVCC= /usr/local/cuda-8.0/bin/nvcc
opt= -O2 -g -G
ARCH= -arch=sm_30
LIBS=
execname= matr_mult

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) -lcurand

matr_mult.o: matr_mult.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c matr_mult.cu

clean:
	rm $(objects)

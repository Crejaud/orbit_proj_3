objects= matr_mult.o kernels.o
NVCC= nvcc
opt= -O2 -g -G
ARCH= -arch=sm_30
LIBS=
execname= matr_mult

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS)

max_elem.o: matr_mult.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c matr_mult.cu

clean:
	rm $(objects)

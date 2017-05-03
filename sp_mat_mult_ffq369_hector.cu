//compile: > nvcc -Xcompiler -Wall -o kern sp_mat_mult_ffq369_hector.cu -DCUDA=1
#include <cstdio>
#include <vector>
#include <cstdlib>
#define LINE_LEN 256
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
//#define imin(a,b) (a<b?a:b)
//const int N=37*1024;
int tpb = 1024;
int bpg = 1024;//imin(32, (N+tpb-1)/tpb);
//Begin CUDA Kernel
__global__ void multiply_kernel(int nrows, int* d_ia, int* d_ja, 
                                double* d_data, double* d_in, double* d_out)
{
   int num_threads = blockDim.x * gridDim.x;
   int tid =  threadIdx.x + blockIdx.x * blockDim.x;
   int mod =  nrows % num_threads;   
   int size = nrows / num_threads;
   int idx = tid * size;
     if(tid < mod){
	size++;
	idx += tid;
     }else 
	idx += mod;
    for(int row = idx; row < (idx+size);row++)
     {
	d_out[row] =0;
        for (int col_idx = d_ia[row]; col_idx < d_ia[row+1]; col_idx++)
        {
           int col = d_ja[col_idx];
	   d_out[row] += d_data[col_idx] * d_in[col];
        }
     }
 
}
// read in a matrix file and populate the compressed-row vectors
int read_mat(const char* name, std::vector<int>& ia, std::vector<int>& ja, std::vector<double>& data)
{
    char line[LINE_LEN];
    FILE* in = fopen(name, "r");

    if (in == NULL)
    {
        return 0;
    }
        
    fgets(line, LINE_LEN, in);  // dummy
    fgets(line, LINE_LEN, in);  // dimension

    int dim = atoi(line);
    // fprintf(stderr, "dimension: %d\n", dim);

    fgets(line, LINE_LEN, in);  // dummy
    fgets(line, LINE_LEN, in);  // nnz

    int nnz = atoi(line);
    // fprintf(stderr, "nnz: %d\n", nnz);

    fgets(line, LINE_LEN, in);  // header

    ia.resize(dim+1);
    ja.resize(nnz);
    data.resize(nnz);


    for (int i=0; i<=dim; i++)
    {
        fgets(line, LINE_LEN, in);
        int idx = atoi(line);
        ia[i]=idx-1;            // data starts with 1
    }


    fgets(line, LINE_LEN, in);  // header
    for (int i=0; i<nnz; i++)
    {
        fgets(line, LINE_LEN, in);
        int idx = atoi(line);
        ja[i]=idx-1;
    }


    fgets(line, LINE_LEN, in);  // header
    for (int i=0; i<nnz; i++)
    {
        fscanf(in, "%lf", &data[i]);
    }

    return 1;
}

// serial matrix/vector multiplication
// 'ia','ja' and 'data' describe the matrix
// 'in' is the vector for the product
// 'out' stores the resulting vector
// 'nrows' is the dimension of the problem
#ifndef CUDA
   #define CUDA_IF(x) if(false)
#else
   #define CUDA_IF(x) if(true)
#endif


void matvec_multiply(int nrows, int nnz, int* ia, int* ja, double* data, double* in, double* out)
{

   #if(false)//serial
   {
     for (int row = 0; row<nrows; row++)
      {
        out[row] = 0;

        for (int col_idx = ia[row]; col_idx < ia[row+1]; col_idx++)
        {
            int col = ja[col_idx];
            out[row] += data[col_idx] * in[col];
        }

      }
   }
   #else
   {
      int *d_ia, *d_ja;
      double *d_data, *d_in, *d_out;
 
      printf("nnz= %d \n", nnz);     
      gpuErrchk(cudaMalloc(&d_in, nrows*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_out, nrows*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_ia, (nrows+1)*sizeof(int)));
      gpuErrchk(cudaMalloc(&d_ja, nnz*sizeof(int)));
      gpuErrchk(cudaMalloc(&d_data, nnz*sizeof(double)));
       
      gpuErrchk(cudaMemcpy(d_in, in, nrows*sizeof(double), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_data, data, nnz*sizeof(double), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_ia, ia, (nrows+1)*sizeof(int), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(d_ja, ja, nnz*sizeof(int), cudaMemcpyHostToDevice));
  
      multiply_kernel<<<bpg,tpb>>>(nrows, d_ia, d_ja, d_data, d_in, d_out);
      
      // copy output result to host here
      gpuErrchk(cudaMemcpy(out, d_out, nrows*sizeof(double), cudaMemcpyDeviceToHost));
      double temp =0;
      for(int i=0; i<nrows; i++)
      {
         temp += out[i];
      }
    // do something more interesting with the results here?
      cudaFree(d_ia); cudaFree(d_ja); cudaFree(d_data); cudaFree(d_in); cudaFree(d_out);
      printf("\nThe result is : %lf\n" ,temp);
    }
    #endif
}

int main(int argc, char *argv[])
{
    std::vector<int>    ia;
    std::vector<int>    ja;
    std::vector<double> data;
    std::vector<double> in;
    std::vector<double> out;
 
    int nrows;
    int nnz;
    if (! read_mat(argv[1], ia, ja, data))
    {
        printf("error reading file\n");
    }
    if(argc > 2)
      tpb = atoi(argv[2]);
    if(argc > 3)
      bpg = atoi(argv[3]);

    	
    // don't forget:  ia.size() is nrows + 1
    nnz = ja.size();
    nrows = ia.size() - 1;
    in.resize(nrows, 1);
    out.resize(nrows);  
    
    matvec_multiply(nrows, nnz, ia.data(), ja.data(), data.data(), in.data(), out.data());
        
    return 0;
}


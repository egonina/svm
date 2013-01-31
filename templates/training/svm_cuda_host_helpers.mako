// GPU data structure pointers
float* devData;
float* devTransposedData;
size_t devTransposedDataPitch;
float* devLabels;
float* devKernelDiag;
float* devAlpha;
float* devF;
void* devResult;

// GPU Cache
float* devCache;
size_t cachePitch;
int devCachePitchInFloats;

//helper data structures
float* devLocalFsRL;
float* devLocalFsRH;
int* devLocalIndicesRL;
int* devLocalIndicesRH;
float* devLocalObjsMaxObj;
int* devLocalIndicesMaxObj;
size_t rowPitch;


// For now, assume data, labels and alphas are passed from Python
// all other data structures are internal...
// ========================== GPU Data Allocation Functions ==========================
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
   }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);            \

void align_host_data(int nPoints, int nDimension) {
    hostPitchInFloats = nPoints;
    if (devDataPitch == nPoints * sizeof(float)) {
      printf("Data is already aligned\n");
      hostData = data;
      hostDataAlloced = false;
    } else {
      hostPitchInFloats = devDataPitch/sizeof(float);	
      hostData = (float*)malloc(devDataPitch*nDimension);
      hostDataAlloced = true;
      printf("Realigning data to a pitch of %i floats\n", hostPitchInFloats);
      for(int i=nDimension-1;i>=0;i--)
        {
          for(int j=nPoints-1;j>=0;j--)
            {
              hostData[i*hostPitchInFloats+j]=data[i*nPoints+j];
            }
        }
    }
}

void alloc_transpose_point_data_on_CPU(int nPoints, int nDimension) {
    // Transpose training data on the CPU
    if (transposedData == 0) {
        transposedData = (float*)malloc(sizeof(float) * nPoints * nDimension);
        transposedDataAlloced = true;
        for(int i = 0; i < nPoints; i++) {
            for (int j = 0; j < nDimension; j++) {
                transposedData[i*nDimension + j] = hostData[j*hostPitchInFloats + i];
            }
        }
    } else {
        transposedDataAlloced = false;
    }
}

void alloc_transposed_point_data_on_GPU(int nPoints, int nDimension) {
    printf("alloc transposed data on GPU\n");
    // Allocate transposed training data on the GPU
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devTransposedData,
                   &devTransposedDataPitch,
                   nDimension*sizeof(float),
                   nPoints));
    CUT_CHECK_ERROR("Alloc transposed point data on GPU failed: ");
}

void copy_transposed_point_data_CPU_to_GPU(int nPoints, int nDimension) {
    // Copy transposed training data to the GPU
    cudaMemcpy2D(devTransposedData,
                   devTransposedDataPitch,
                   transposedData,
                   nDimension*sizeof(float),
                   nDimension*sizeof(float),
      			   nPoints,
                   cudaMemcpyHostToDevice);
    CUT_CHECK_ERROR("Copy transposed point data from CPU to GPU failed: ");
}

void alloc_point_data_on_GPU(int nPoints, int nDimension) {
    // Allocate training data on the GPU
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devData, &devDataPitch, nPoints*sizeof(float), nDimension));
    CUT_CHECK_ERROR("Alloc point data on GPU failed: ");
    align_host_data(nPoints, nDimension);
    alloc_transpose_point_data_on_CPU(nPoints, nDimension);
    alloc_transposed_point_data_on_GPU(nPoints, nDimension);
    copy_transposed_point_data_CPU_to_GPU(nPoints, nDimension);
}

void copy_point_data_CPU_to_GPU(int nDimension) {
    // Copy the training data to the GPU
    CUDA_SAFE_CALL(cudaMemcpy(devData, hostData, devDataPitch*nDimension, cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy point data from CPU to GPU failed: ");
}

void alloc_labels_on_GPU(int nPoints) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLabels, nPoints*sizeof(float)));
    CUT_CHECK_ERROR("Alloc labels on GPU failed: ");
}

void copy_labels_CPU_to_GPU(int nPoints) {
    CUDA_SAFE_CALL(cudaMemcpy(devLabels, labels, nPoints*sizeof(float), cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("Copy labels from CPU to GPU failed: ");
}

void alloc_kernel_diag_on_GPU(int nPoints) {
    // Allocate kernel diagonal elements (?) on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&devKernelDiag, nPoints*sizeof(float)));
    CUT_CHECK_ERROR("Alloc kernelDiag on GPU failed: ");
}
    
void alloc_alphas_on_GPU(int nPoints) {
    // Allocate support vectors on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&devAlpha, nPoints*sizeof(float)));
    CUT_CHECK_ERROR("Alloc alphas on GPU failed: ");
}
    
void alloc_F_on_GPU(int nPoints) {
    // Allocate the error array on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&devF, nPoints*sizeof(float)));
    CUT_CHECK_ERROR("Alloc F on GPU failed: ");
}
    
void alloc_result_on_GPU() {
    CUDA_SAFE_CALL(cudaMalloc(&devResult, 8*sizeof(float)));
    CUT_CHECK_ERROR("Alloc result on GPU failed: ");
}

void alloc_helper_structures_on_GPU(int blockWidth) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRL, blockWidth*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalFsRH, blockWidth*sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRL, blockWidth*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesRH, blockWidth*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalObjsMaxObj, blockWidth*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devLocalIndicesMaxObj, blockWidth*sizeof(int)));
    CUT_CHECK_ERROR("Alloc helper structures on GPU failed: ");
}

void compute_row_pitch(int nPoints) {
    void* temp;
    CUDA_SAFE_CALL(cudaMallocPitch(&temp, &rowPitch, nPoints*sizeof(float), 2));
    CUDA_SAFE_CALL(cudaFree(temp));
    CUT_CHECK_ERROR("Compute row pitch on GPU failed: ");
}

void allocate_GPU_cache(int nPoints) {
    // Determine the size of the cachg
    size_t remainingMemory;
    size_t totalMemory;
    cuMemGetInfo(&remainingMemory, &totalMemory);
    
    int sizeOfCache = remainingMemory/((int)rowPitch);
    sizeOfCache = (int)((float)sizeOfCache*0.95);//If I try to grab all the memory available, it'll fail
    if (nPoints < sizeOfCache) {
    	sizeOfCache = nPoints;
    }
      	
    printf("%zd bytes of memory found on device, %zd bytes currently free\n", totalMemory, remainingMemory);
    printf("%d rows of kernel matrix will be cached (%d bytes per row)\n", sizeOfCache, (int)rowPitch);
    
    // Allocate cache on the GPU
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&devCache, &cachePitch, nPoints*sizeof(float), sizeOfCache));
    CUT_CHECK_ERROR("Alloc cache on GPU failed: ");
    devCachePitchInFloats = (int)cachePitch/(sizeof(float));
}

void dealloc_point_data_on_GPU(){
    cudaFree(devData);
    CUT_CHECK_ERROR("Dealloc point data on GPU failed: ");
}

void dealloc_transposed_point_data_on_GPU(){
    cudaFree(devTransposedData);
    CUT_CHECK_ERROR("Dealloc transposed point data on GPU failed: ");
}

void dealloc_labels_on_GPU(){
    cudaFree(devLabels);
    CUT_CHECK_ERROR("Dealloc labels on GPU failed: ");
}

void dealloc_alphas_on_GPU(){
    cudaFree(devAlpha);
    CUT_CHECK_ERROR("Dealloc alphas on GPU failed: ");
}

void dealloc_devF_on_GPU(){
    cudaFree(devF);
    CUT_CHECK_ERROR("Dealloc F on GPU failed: ");
}

void dealloc_devCache_on_GPU(){
    cudaFree(devCache);
    CUT_CHECK_ERROR("Dealloc cache on GPU failed: ");
}

void dealloc_helper_structures_on_GPU(){
    cudaFree(devLocalIndicesRL);
    cudaFree(devLocalIndicesRH);
    cudaFree(devLocalFsRH);
    cudaFree(devLocalFsRL);
    cudaFree(devKernelDiag);
    cudaFree(devResult);
    cudaFree(devLocalIndicesMaxObj);
    cudaFree(devLocalObjsMaxObj);
    CUT_CHECK_ERROR("Dealloc helper structures on GPU failed: ");
}

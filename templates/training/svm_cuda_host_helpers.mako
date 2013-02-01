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

// Keep the following two functions here, called internally when
// allocating data on the GPU
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

void alloc_transposed_point_data_on_CPU(int nPoints, int nDimension) {
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
    alloc_transposed_point_data_on_CPU(nPoints, nDimension);
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

void alloc_alphas_on_GPU(int nPoints) {
    // Allocate support vectors on the GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&devAlpha, nPoints*sizeof(float)));
    CUT_CHECK_ERROR("Alloc alphas on GPU failed: ");
}
    
void alloc_result_on_GPU() {
    CUDA_SAFE_CALL(cudaMalloc(&devResult, 8*sizeof(float)));
    CUT_CHECK_ERROR("Alloc result on GPU failed: ");
}

void dealloc_transposed_point_data_on_CPU() {
    if (transposedDataAlloced) {
      free(transposedData);
    }
}

void dealloc_transposed_point_data_on_GPU(){
    cudaFree(devTransposedData);
    CUT_CHECK_ERROR("Dealloc transposed point data on GPU failed: ");
}

void dealloc_host_data_on_CPU() {
    if (hostDataAlloced) {
      free(hostData);
    }
}

void dealloc_point_data_on_GPU(){
    cudaFree(devData);
    CUT_CHECK_ERROR("Dealloc point data on GPU failed: ");
    dealloc_transposed_point_data_on_CPU();
    dealloc_transposed_point_data_on_GPU();
    dealloc_host_data_on_CPU();
}

void dealloc_labels_on_GPU(){
    cudaFree(devLabels);
    CUT_CHECK_ERROR("Dealloc labels on GPU failed: ");
}

void dealloc_alphas_on_GPU(){
    cudaFree(devAlpha);
    CUT_CHECK_ERROR("Dealloc alphas on GPU failed: ");
}

void dealloc_result_on_GPU(){
    cudaFree(devResult);
    CUT_CHECK_ERROR("Dealloc result on GPU failed: ");
}

void printModel(const char* outputFile, int kernel_type, float gamma, float coef0, float degree, float* alpha, float* labels, float* data, int nPoints, int nDimension, float epsilon) { 

	printf("Output File: %s\n", outputFile);
	FILE* outputFilePointer = fopen(outputFile, "w");
	if (outputFilePointer == NULL) {
		printf("Can't write %s\n", outputFile);
		exit(1);
	}

	int nSV = 0;
	int pSV = 0;
	for(int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			if (labels[i] > 0) {
				pSV++;
			} else {
				nSV++;
			}
		}
	}

  bool printGamma = false;
  bool printCoef0 = false;
  bool printDegree = false;
  if (kernel_type == 2) {
    printGamma = true;
    printCoef0 = true;
    printDegree = true;
  } else if (kernel_type == 1) {
    printGamma = true;
  } else if (kernel_type == 3) {
    printGamma = true;
    printCoef0 = true;
  }
	
	fprintf(outputFilePointer, "svm_type c_svc\n");
	fprintf(outputFilePointer, "kernel_type %d\n", kernel_type);
  if (printDegree) {
    fprintf(outputFilePointer, "degree %f\n", degree);
  }
  if (printGamma) {
    fprintf(outputFilePointer, "gamma %f\n", gamma);
  }
  if (printCoef0) {
    fprintf(outputFilePointer, "coef0 %f\n", coef0);
  }
	fprintf(outputFilePointer, "nr_class 2\n");
	fprintf(outputFilePointer, "total_sv %d\n", nSV + pSV);
	//fprintf(outputFilePointer, "rho %.10f\n", kp.b);
	fprintf(outputFilePointer, "label 1 -1\n");
	fprintf(outputFilePointer, "nr_sv %d %d\n", pSV, nSV);
	fprintf(outputFilePointer, "SV\n");
	for (int i = 0; i < nPoints; i++) {
		if (alpha[i] > epsilon) {
			fprintf(outputFilePointer, "%.10f ", labels[i]*alpha[i]);
			for (int j = 0; j < nDimension; j++) {
				fprintf(outputFilePointer, "%d:%.10f ", j+1, data[j*nPoints + i]);
			}
			fprintf(outputFilePointer, "\n");
		}
	}
	fclose(outputFilePointer);
}


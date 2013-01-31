// ======================== SVM TRAIN ================
void train(int nPoints, int nDimension,
           int kernel_type, float gamma,
           float coef0, float degree,
           float cost, int heuristicMethod,
           float epsilon, float tolerance) {
    
    float cEpsilon = cost - epsilon;
    Controller progress(2.0, heuristicMethod, 64, nPoints);

    // Determine kernel type and parameters
    int kType = GAUSSIAN;
    float parameterA;
    float parameterB;
    float parameterC;
    if (kernel_type == 1) {
        parameterA = -gamma;
        kType = GAUSSIAN;
        printf("Gaussian kernel: gamma = %f\n", -parameterA);
    } else if (kernel_type = 2) {
        parameterA = gamma;
        parameterB = coef0;
        parameterC = degree;
        kType = POLYNOMIAL;
        printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    } else if (kernel_type == 0) {
        kType = LINEAR;
        printf("Linear kernel\n");
    } else if (kernel_type == 3) {
        kType = SIGMOID;
        parameterA = gamma;
        parameterB = coef0;
        printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
        if ((parameterA <= 0) || (parameterB < 0)) {
            printf("Invalid Parameters\n");
            exit(1);
        }
    }
    printf("Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon);
    
    Cache kernelCache(nPoints, sizeOfCache);
    
    // Set number of blocks and number of threads per block
    dim3 threadsLinear(${num_threads});
    dim3 blocksLinear(${num_blocks});
    
    int devDataPitchInFloats = ((int)devDataPitch) >> 2;
    int devTransposedDataPitchInFloats = ((int)devTransposedDataPitch) >> 2;
      
    // Initialize
    launchInitialization(devData, devDataPitchInFloats, nPoints, nDimension, kType, parameterA, parameterB, parameterC, devKernelDiag, devAlpha, devF, devLabels, blocksLinear, threadsLinear);
    cudaError_t err = cudaGetLastError();
    if(err) printf("Error: %s\n", cudaGetErrorString(err));
    printf("Initialization complete\n");
    
    //Choose initial points
    float bLow = 1;
    float bHigh = -1;
    int iteration = 0;
    int iLow = -1;
    int iHigh = -1;
    for (int i = 0; i < nPoints; i++) {
    	if (labels[i] < 0) {
    		if (iLow == -1) {
    			iLow = i;
    			if (iHigh > -1) {
    				i = nPoints; //Terminate
    			}
    		}
    	} else {
    		if (iHigh == -1) {
    			iHigh = i;
    			if (iLow > -1) {
    				i = nPoints; //Terminate
    			}
    		}
    	}
    }
    
    dim3 singletonThreads(1);
    dim3 singletonBlocks(1);
    launchTakeFirstStep(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow, iHigh, kType, parameterA, parameterB, parameterC, singletonBlocks, singletonThreads);
    CUDA_SAFE_CALL(cudaMemcpy((void*)hostResult, devResult, 8*sizeof(float), cudaMemcpyDeviceToHost));
    
    
    float alpha2Old = *(hostResult + 0);
    float alpha1Old = *(hostResult + 1);
    bLow = *(hostResult + 2);
    bHigh = *(hostResult + 3);
    float alpha2New = *(hostResult + 6);
    float alpha1New = *(hostResult + 7);
    
    float alpha1Diff = alpha1New - alpha1Old;
    float alpha2Diff = alpha2New - alpha2Old;
    
    int iLowCacheIndex;
    int iHighCacheIndex;
    bool iLowCompute;
    bool iHighCompute; 
    
    dim3 reduceThreads(${num_threads});
      
    for (iteration = 1; true; iteration++) {
    
    	if (bLow <= bHigh + 2*tolerance) {
    		break; //Convergence!!
    	}
    
    	if ((iteration & 0x7ff) == 0) {
    		printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
    	}
    
        if ((iteration & 0x7f) == 0) {
            heuristicMethod = progress.getMethod();
        }
        
	    kernelCache.findData(iHigh, iHighCacheIndex, iHighCompute);
   	    kernelCache.findData(iLow, iLowCacheIndex, iLowCompute);
        
        if (heuristicMethod == FIRSTORDER) {
            launchFirstOrder(iLowCompute, iHighCompute,
                             kType, nPoints, nDimension,
                             blocksLinear, threadsLinear,
                             reduceThreads, devData,
                             devDataPitchInFloats, devTransposedData,
                             devTransposedDataPitchInFloats, devLabels,
                             epsilon, cEpsilon, devAlpha, devF,
                             alpha1Diff * labels[iHigh],
                             alpha2Diff * labels[iLow], iLow, iHigh,
                             parameterA, parameterB, parameterC,
                             devCache, devCachePitchInFloats,
                             iLowCacheIndex, iHighCacheIndex,
                             devLocalIndicesRL, devLocalIndicesRH,
                             devLocalFsRH, devLocalFsRL,
                             devKernelDiag, devResult, cost);
          } else {
            launchSecondOrder(iLowCompute, iHighCompute,
                              kType, nPoints, nDimension,
                              blocksLinear, threadsLinear,
                              reduceThreads, devData,
                              devDataPitchInFloats, devTransposedData,
                              devTransposedDataPitchInFloats, devLabels,
                              epsilon, cEpsilon, devAlpha, devF,
                              alpha1Diff * labels[iHigh],
                              alpha2Diff * labels[iLow], iLow, iHigh,
                              parameterA, parameterB, parameterC,
                              &kernelCache, devCache, devCachePitchInFloats,
                              iLowCacheIndex, iHighCacheIndex,
                              devLocalIndicesRH, devLocalFsRH, devLocalFsRL,
                              devLocalIndicesMaxObj, devLocalObjsMaxObj,
                              devKernelDiag, devResult,
                              hostResult, cost, iteration);
            //cudaError_t err = cudaGetLastError();
            //if(err) printf("Error: %s\n", cudaGetErrorString(err));
          }
          //printf("iteration: %i\n", iteration);
          CUDA_SAFE_CALL(cudaMemcpy((void*)hostResult, devResult, 8*sizeof(float), cudaMemcpyDeviceToHost));
        
          alpha2Old = *(hostResult + 0);
          alpha1Old = *(hostResult + 1);
          bLow = *(hostResult + 2);
          bHigh = *(hostResult + 3);
          iLow = *((int*)hostResult + 6);
          iHigh = *((int*)hostResult + 7);
          alpha2New = *(hostResult + 4);
          alpha1New = *(hostResult + 5);
          alpha1Diff = alpha1New - alpha1Old;
          alpha2Diff = alpha2New - alpha2Old;
          progress.addIteration(bLow-bHigh);
          CUT_CHECK_ERROR("SMO Iteration Failed");
    }
      
    printf("%d iterations\n", iteration);
    printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
    //kp->b = (bLow + bHigh) / 2;
    float b = (bLow + bHigh) / 2;
    kernelCache.printStatistics();
    CUDA_SAFE_CALL(cudaMemcpy((void*)alpha, devAlpha, nPoints*sizeof(float), cudaMemcpyDeviceToHost));
}

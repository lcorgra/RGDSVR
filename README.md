# RGDSVR
Tools for Robust Generative Diffeomorphic Slice to Volume Reconstructions (RGDSVR)

This repository provides tools to implement the methods in the manuscript ''Fetal MRI by robust deep generative prior reconstruction and diffeomorphic registration: application to gestational age prediction'', L Cordero-Grande, JE Ortuño-Fisac, A Uus, M Deprez, A Santos, JV Hajnal, and MJ Ledesma-Carbayo, arXiv, 2021.

The code has been developed in MATLAB and has the following structure:

###### ./
contains a script to run a reconstruction of the provided example data: *rgdsvr_example.m* and another to import the Python code *loadPythonDeepFetal.m*.

###### ./SVR
contains files to perform SVR reconstructions: *svrAlternateMinimization.m*, *svrCG.m*, *svrDD.m*, *svrDecode.m*, *svrEncode.m*, *svrExcitationStructures.m*, *svrRearrangeAxes.m*, *svrSetUp.m*, *svrSliceWeights.m*, *svrSolveDPack.m*, *svrSolveDVolu.m*, *svrSolveTVolu.m*.

###### ./SVR/Common
contains common functions used by SVR methods: *computeDeformableTransforms.m*, *finalizeConvergenceControl.m*, *initializeConvergenceControl.m*, *initializeDEstimation.m*, *modulateGradient.m*, *prepareLineSearch.m*, *updateRule.m*.

###### ./Alignment
contains functions for registration.

###### ./Alignment/Elastic
contains functions for elastic registration: *adAdjointOperator.m*, *adDualOperator.m*, *buildDifferentialOperator.m*, *buildGradientOperator.m*, *buildMapSpace.m*, *computeGradientHessianElastic.m*, *computeJacobian.m*, *computeRiemannianMetric.m*, *deformationGradientTensor.m*, *deformationGradientTensorSpace.m*, *elasticTransform.m*, *geodesicShooting.m*, *integrateReducedAdjointJacobi.m*, *integrateVelocityFields.m*, *invertElasticTransform.m*, *mapSpace.m*, *precomputeFactorsElasticTransform.m*.

###### ./Alignment/Metrics
contains functions for metrics used in registration: *computeMetricDerivativeHessianRigid.m*, *metricFiltering.m*, *metricMasking.m*, *msdMetric.m*.

###### ./Alignment/Rigid
contains functions for rigid registration: *convertRotation.m*, *factorizeHomogeneousMatrix.m*, *generatePrincipalAxesRotations.m*, *generateTransformGrids.m*, *jacobianQuaternionEuler.m*, *jacobianShearQuaternion.m*, *mapVolume.m*, *modifyGeometryROI.m*, *precomputeFactorsSincRigidTransformQuick.m*, *quaternionToShear.m*, *restrictTransform.m*, *rotationDistance.m*, *shearQuaternion.m*, *sincRigidTransformGradientQuick.m*, *sincRigidTransformQuick.m*.

###### ./Build
contains functions that replace, extend or adapt some MATLAB built-in functions: *aplGPU.m*, *det2x2m.m*, *det3x3m.m*, *diagm.m*, *dynInd.m*, *eigm.m*, *eultorotm.m*, *gridv.m*, *ind2subV.m*, *indDim.m*, *matfun.m*, *multDimMax.m*, *multDimMin.m*, *multDimSum.m*, *numDims.m*, *parUnaFun.m*, *quattoeul.m*, *resPop.m*, *resSub.m*, *rotmtoquat.m*, *sub2indV.m*, *svdm.m*.

###### ./Control
contains functions to control the implementation and parameters of the algorithm: *channelsDeepDecoder.m*, *parametersDeepDecoder.m*, *svrAlgorithm.m*, *useGPU.m*.

###### ./Methods
contains functions that implement generic methods for reconstruction: *build1DCTM.m*, *build1DFTM.m*, *buildFilter.m*, *buildStandardDCTM.m*, *buildStandardDFTM.m*, *computeROI.m*, *extractROI.m*, *fctGPU.m*, *fftGPU.m*, *filtering.m*, *flipping.m*, *fold.m*, *generateGrid.m*, *ifctGPU.m*, *ifftGPU.m*, *ifold.m*, *mirroring.m*, *resampling.m*.

###### ./Python/deepfetal/deepfetal
contains python methods.

###### ./Python/deepfetal/deepfetal/arch
contains python methods to build deep architectures: *deepdecoder.py*.

###### ./Python/deepfetal/deepfetal/build
contains python methods with generic functions: *bmul.py*, *complex.py*, *dynind.py*, *matcharrays.py*, *shift.py*.

###### ./Python/deepfetal/deepfetal/lay
contains python methods to build deep layers: *encode.py*, *resample.py*, *sinc.py*, *sine.py*, *swish.py*, *tanh.py*.

###### ./Python/deepfetal/deepfetal/meth
contains python methods with generic deep methodologies: *apl.py*, *resampling.py*, *tmtx.py*, *t.py*.

###### ./Python/deepfetal/deepfetal/opt
contains python methods for optimization: *cost.py*, *fit.py*.

###### ./Python/deepfetal/deepfetal/unit
contains python methods to build deep units: *atac.py*  *decoder.py*.

###### ./Tools
contains auxiliary tools: *findString.m*, *removeExtension.m*, *writenii.m*.

###### ./Tools/NIfTI_20140122
from https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image


NOTE 1: Example data provided in the dataset *svr_inp_034.mat*. For runs without changing the paths, it should be placed in folder
###### ../RGDSVR-Data
Data generated when running the example script appears in this folder with names *svr_out_034.mat* and *x_034.mat*.

NOTE 2: Instructions for linking the python code in *loadPythonDeepFetal.m*.

NOTE 3: *pathAnaconda* variable in *rgdsvr_example.m* needs to point to parent of python environment.

NOTE 4: Example reconstruction takes about half an hour in a system equipped with a GPU NVIDIA GeForce RTX 3090.

function svr=svrAlgorithm(svr)

%SVRALGORITHM   Sets the algorithm parameters
%   SVR=SVRALGORITHM(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure with algorithm parameters
%

%CONTROL
svr.Alg.Debug=1;%To show debug info

%FOV DEFINITION  
svr.Alg.StackGeometry=1;%Stack used to derive the geometry of the reconstruction   
svr.Alg.Windowing=[0.3 0 0];%Apodization in the slice direction / Apodization of reconstructed FOV --- not very important
svr.Alg.Resol=min(min(cat(3,svr.MSZ{:}),[],3));%Final resolution
svr.Alg.SecurityFOV=16;%Security margin of FOV in mm

%RESOLUTION AND SLICE PROFILE
%svr.Alg.FWHM=svr.ParZ{1}.SlTh/min(min(cat(3,svr.MSZ{:}),[],3));%Physical slice profile PSF ratio    
svr.Alg.FWHM=3;%Computational slice profile PSF ratio, a bit bigger than physical could model deviations from ideal when spins are in motion
svr.Alg.MS=1.25;%Resolution of the reconstruction
svr.Alg.Resol=round(16*svr.Alg.Resol/svr.Alg.MS)*svr.Alg.MS/16;%Modification for rounding issues

%ROBUST RECONSTRUCTION
svr.Alg.RobustReconstruction=1;%1 to use robust reconstruction / 0 not to use robust reconstruction
svr.Alg.NItMax=256;%Maximum number of iterations of correction, never reached in experiments
if svr.Alg.RobustReconstruction;svr.Alg.Lp=0;else svr.Alg.Lp=2-1e-12;end%Norm of the reconstruction (to suppress outliered slices)
svr.Alg.Block=5;%Block size for outlier rejection, related to width of Gaussian kernel
svr.Alg.TuningConstant=sqrt(2);%Lower, more outlier rejection, higher, less outlier rejection
svr.Alg.RemoveSlicesFactor=0.1;%Factor to remove slices from computation in case they transform to the background
svr.Alg.NC=5;%Fixed number of iterations

%MOTION CORRECTION
svr.Alg.MotionCorrectedReconstruction=1;%1 to use motion correction / 0 not to use motion correction
svr.Alg.Nl=5;%Full reconstruction, all levels,  0-> no correction / 1-> warming up / 2-> per-stack rigid correction / 3-> per-stack elastic correction / 4-> per-package elastic correction / 5-> per-interleave elastic correction
svr.Alg.ConvL=1;%Increase to accelerate convergence for T
svr.Alg.RobustMotion=0;%To estimate motion based on robust information (enabling would be experimental)

%RIGID
svr.Alg.Winit=[1e-2 1];%Damping parameter for LM optimizers, first for rigid, second for deformable
svr.Alg.MotJointFracOrd=0;%To use joint fractional finite difference
svr.Alg.MotFracOrd=[0 0];%Fractional-finite-difference derivative order of the metric, first for rigid, second for deformable
svr.Alg.NG=1;%Number of steps for velocity integration (LDMM)
svr.Alg.NTRes=8;%Resolution of diffeomorphism (mm). This needs to be set in agreement with RegDiff so that inversion does not introduce artifacts, increase if RegDiff is reduced
if svr.Alg.MotionCorrectedReconstruction;svr.Alg.InvVar=1e-1*ones(1,3);else svr.Alg.InvVar=1e5*ones(1,3);end%Inverse of variance for regularization of registration
svr.Alg.RegDiff=[2.5 2.5];%Regularization of diffeomorphism, first is ratio of differential operator versus Tikhonov and second is exponent (this is squared)
svr.Alg.NconvT=2;%Number of iterations below a given level of motion to have converged
svr.Alg.NItMaxD=128;%Maximum number of iterations for parameter optimization
svr.Alg.Ups=[0;1;1;2];%Upsampling factors for circular convolutions and interpolation (diffeomorphisms/ gradient / transform/ inversion)
svr.Alg.BaseConvLevel=0.5*ones(1,3);%Base convergence level as a percentage of the pixel size respectively for deformable stack / package / interleave
svr.Alg.ConstrainDiffeo=1;%To constrain to diffeomorphic transformations
svr.Alg.ThInvert=[0.4 0.6 0.8];%Threshold to modulate the gradient depending on expected invertibility of field (0.85 matches that for constraining to diffeomorphic transformations), first stack, second package, third interleave

%REGULARIZATION
svr.Alg.RegularizationType=2;%0 no regularization / 1 basic regularization / 2 deep decoder regularization
if ismember(svr.Alg.RegularizationType,[0 2])
    svr.Alg.RegFracOrd=0;%Fractional order of different regularization terms
    svr.Alg.Ti=2.^-8;%Minimum regularization weight
elseif svr.Alg.RegularizationType==1
    svr.Alg.RegFracOrd=[0 1 2 4 8 16 32];%Fractional order of different regularization terms
    svr.Alg.Ti=2.^(-8:-2);%Regularization weight
end
svr.Alg.Lambda=0.4;%Regularization weight deep decoder
svr.Alg.Epochs=180;%Base epochs for deep decoder

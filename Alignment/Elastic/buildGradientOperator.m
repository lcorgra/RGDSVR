function F=buildGradientOperator(T,N,spGrid,gpu)

%BUILDGRADIENTOPERATOR   Builds a gradient operator for [1] M Zhang, PT 
%Fletcher, "Fast Diffeomorphic Image Registration via Fourier-Approximated 
%Lie Algebras," 127:61-73, 2019
%   F=BUILDGRADIENTOPERATOR(T,N,SPGRID,{GPU})
%   * {T} is the type, one of the following '1stFiniteDiscreteForward' / 
%   '1stFiniteDiscreteBackward' / '1stFiniteDiscreteCentered' (default) or
%   if a cell, the filters along the different dimensions
%   * N are the dimensions of the operator
%   * SPGRID is the spacing of the spatial grid
%   * {GPU} determines whether to use gpu computations 
%   ** F is the operator
%

if nargin<4 || isempty(gpu);gpu=useGPU;end
ND=length(spGrid);

if ~iscell(T)%It is the type of filter, one of the following '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / '1stFiniteDiscreteCentered'
    %T='1stFinite';%HARDCODED TO MAKE PROBLEM INDEPENDENT OF
    %RESOUTION---BAD IDEA BECAUSE WE WANT IT DISCRETE
    F=cell(1,ND);
    for n=1:ND
        M=N(1:ND);M(setdiff(1:ND,n))=1;             
        F{n}=buildFilter(M,T,spGrid,0,1);
        if gpu;F{n}=gpuArray(F{n});end
    end
else%The filter is already constructed, do nothing
    F=T;
end

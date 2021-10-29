function L=buildDifferentialOperator(L,N,spGrid,gpu,c)

%BUILDDIFFERENTIALOPERATOR   Builds a differential operator following [1] 
%M Zhang, PT Fletcher, "Fast Diffeomorphic Image Registration via 
%Fourier-Approximated Lie Algebras," 127:61-73, 2019
%   L=BUILDDIFFERENTIALOPERATOR(L,N,SPGRID,{GPU},{C})
%   * L could be parameters of the operator
%   * N are the dimensions of the operator
%   * SPGRID is the spacing of the spatial grid
%   * {GPU} determines whether to use gpu computations 
%   * {C} serves to use the cosine transform 
%   ** L is the operator
%

if nargin<4 || isempty(gpu);gpu=useGPU;end
if nargin<5 || isempty(c);c=0;end%Use cosine transform

ND=length(spGrid);

if numel(L)==2%We assume these are alpha and c of L=(-2*alpha*(sum_d^D(cos(2*pi*k_d/N_d))-D)+1)^c, possible values alpha=3, c=3 
    A=1;
    for n=1:ND
        M=(c+1)*N(1:ND);M(setdiff(1:ND,n))=1;             
        F=real(-L(1)*buildFilter(M,'2ndFiniteDiscrete',1,0,2));%TO MATCH FLASH IMPLEMENTATION        
        %F=real(-L(1)*buildFilter(M,'2ndFiniteDiscrete',spGrid,0,2));        
        %F=real(-L(1)*buildFilter(M,'2ndFinite',spGrid,0,2));%TO MAKE PROBLEM INDEPENDENT OF RESOLUTION
        if c==1;F(N(n)+1:2*N(n))=[];end
        if gpu;F=gpuArray(F);end
        A=bsxfun(@plus,A,F);
        %figure;plot(A(:))
        %pause
    end
    L=A.^(L(2));
    L(L>1000000)=1000000;
end

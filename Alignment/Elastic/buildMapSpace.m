function F=buildMapSpace(x,di,M,N)

%BUILDMAPSPACE   Precomputes factors to map from Fourier to image or image 
%to Fourier spaces for implementing [1] M Zhang, PT Fletcher, "Fast 
%Diffeomorphic Image Registration via Fourier-Approximated Lie Algebras," 
%127:61-73, 2019. These are applied in function mapSpace
%   F=BUILDMAPSPACE(X,DI,M,N)
%   * X is a field
%   * DI is the mapping, 0 to image space / 1 to Fourier space
%   * M are the dimensions in image space/Image to Fourier space matrices
%   if cell
%   * N are the dimensions in Fourier space/Fourier to image space matrices
%   if cell
%   ** F is a Fourier / inverse Fourier transformation
%

ND=min(length(N),length(M));

gpu=isa(x,'gpuArray');
F=cell(1,3);
if di==0
    for n=1:ND
        [~,FF]=build1DFTM(M(n),0,gpu);
        F{n}=FF;
        if (gpu && isaUnderlying(x,'double')) || isa(x,'double');F{n}=double(F{n});end
        F{n}=(M(n)*sqrt(M(n)/N(n)))*resampling(F{n},[M(n) N(n)],1);
    end
else
    for n=1:ND
        FF=build1DFTM(M(n),0,gpu);
        F{n}=FF;
        if (gpu && isaUnderlying(x,'double')) || isa(x,'double');F{n}=double(F{n});end
        F{n}=resampling(F{n},[N(n) M(n)],1)/(M(n)*sqrt(N(n)/M(n)));   
    end
end

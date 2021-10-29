function x=mapSpace(x,di,M,N)

%MAPSPACE   Maps from Fourier to image or image to Fourier spaces for 
%implementing [1] M Zhang, PT Fletcher, "Fast Diffeomorphic Image 
%Registration via Fourier-Approximated Lie Algebras," 127:61-73, 2019
%   X=MAPSPACE(X,DI,M,N)
%   * X is a field
%   * DI is the mapping, 0 to image space / 1 to Fourier space
%   * M are the dimensions in image space/Image to Fourier space matrices
%   if cell
%   * N are the dimensions in Fourier space/Fourier to image space matrices
%   if cell
%   ** X is the field in the new domain
%

NDD=min(length(M),length(N));
ND=numDims(x);NV=numel(x);
if ~iscell(M)
    if di;F=buildMapSpace(x,1,M,N);%Think we assume M is bigger than N
    else FH=buildMapSpace(x,0,M,N);
    end
else
    F=M;FH=N;
    M=zeros(1,NDD);N=zeros(1,NDD);
    for m=1:NDD;M(m)=size(F{m},1);N(m)=size(F{m},2);end
end

    
if di==0%Map to image space
    if ND==1;x=reshape(x,[N NV/prod(N)]);end
    %for m=NDD:-1:1;x=ifftGPU(x,m,FH{m});end%Simply because ifft allows to change size            
    for m=NDD:-1:1;x=aplGPU(FH{m},x,m);end
    x=real(x);   
    if ND==1;x=x(:);end
else%Map to Fourier space  
    if ND==1;x=reshape(x,[M NV/prod(M)]);end
    %for m=1:NDD;x=ifftGPU(x,m,F{m});end%Simply because ifft allows to change size
    for m=1:NDD;x=aplGPU(F{m},x,m);end
    if ND==1;x=x(:);end
end

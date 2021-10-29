function svr=svrSliceWeights(svr,non)

%SVRSLICEWEIGHTS   Computes the reliability of each slice
%   SVR=SVRSLICEWEIGHTS(SVR,NON)
%   * SVR is a svr structure
%   * SVR is a svr structure after computing the weights
%

gpu=isa(svr.x,'gpuArray');

svr.xx=svr.x;
svr=svrEncode(svr);
blSz=ceil(svr.Alg.Block*(svr.Alg.Resol/svr.Alg.MS));       
if mod(blSz,2)==0;blSz=blSz+1;end
%fprintf('Block size %d\n',blSz);
k=[];

%COMPUTE SMOOTHED RESIDUALS
ROI=cell(1,svr.NV);
ww=cell(1,svr.NV);
for v=1:svr.NV
    y=svr.y{v};
    x=svr.yy{v};                        
    if useGPU;y=gpuArray(y);x=gpuArray(x);end

    xs=multDimSum(abs(x).^2,svr.id(v,1:2));
    xs(xs<0.01*mean(xs))=0;
    x=dynInd(x,xs(:)==0,svr.id(v,3),0);        
    ROI{v}=computeROI(x); 

    or=ones(1,3);
    or(svr.id(v,1:2))=blSz;
    x=extractROI(x,ROI{v});
    y=extractROI(y,ROI{v});
    x=x-y;          
    x=abs(x).^2;                        
    or=ones(1,3);or(svr.id(v,1:2))=blSz;            
    ww{v}=buildFilter(or,'gauss',1,gpu,1);
    for m=1:3;ww{v}=fftshift(ww{v},m);end
    NX=size(x);                                    

    x=padarray(x,(or-1)/2,'symmetric','both');
    NNX=size(x);
    ww{v}=resampling(ww{v},NNX,3);
    ww{v}=ww{v}/sum(ww{v}(:));
    for m=1:3
        ww{v}=ifftshift(ww{v},m);
        ww{v}=fftGPU(ww{v},m);
    end
    x=max(real(filtering(x,ww{v})),0);
    x=resampling(x,NX,3);
    svr.W{v}=x;
    k=cat(1,k,sqrt(x(:)));    
end

%COMPUTE WEIGHTS
s=prctile(k,50);
s=(1.4826*svr.Alg.TuningConstant*s)^2;
for v=1:svr.NV
    svr.W{v}=exp(-svr.W{v}/s);
    or=ones(1,3);or(svr.id(v,1:2))=blSz;   
    NX=size(svr.W{v});
    svr.W{v}=padarray(svr.W{v},(or-1)/2,'symmetric','both');
    svr.W{v}=max(real(filtering(svr.W{v},ww{v})),0);
    svr.W{v}=resampling(svr.W{v},NX,3);
    svr.W{v}=extractROI(svr.W{v},ROI{v},0);
end

%APPLY CURRENT EXPONENT
for v=1:svr.NV
    svr.W{v}=sqrt(svr.W{v}).^(2-non);      
    svr.W{v}=extractROI(extractROI(svr.W{v},ROI{v}),ROI{v},0);%This is extremely important as boundary condition for convergence, otherwise 0^0=1 and it deforms to areas outside the FOV
end
if svr.Alg.Debug;fprintf('Exponent factor weights estimation: %.2f\n',non);end

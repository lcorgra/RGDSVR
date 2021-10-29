function [svr,conv]=svrSolveDPack(svr,typ)

%SVRSOLVEDPACK   Solve for a deformable transform D on a per-package/per-interleave level
%   [SVR,CONV]=SVRSOLVEDPACK(SVR,TYP)
%   * SVR is a svr structure
%   * TYP indicates whether to operate at per-interleave (0) or 
%   per-package (1) level
%   ** SVR is a svr structure after an iteration of solver
%   ** CONV is a flag to indicate convergence
%

profi=0;%For profiling
debug=svr.Alg.Debug;%For seeing info

if typ==0
    if isfield(svr,'DV')
        for v=1:svr.NV;svr.VI{v}=repmat(dynInd(svr.VV,v,5),[ones(1,4) svr.I(v)]);end
        svr=rmfield(svr,{'DV','DVH'});
    end    
elseif typ==1
    if isfield(svr,'DV')
        for v=1:svr.NV;svr.VP{v}=repmat(dynInd(svr.VV,v,5),[ones(1,4) svr.P(v)]);end
        svr=rmfield(svr,{'DV','DVH'});
        svr=rmfield(svr,{'DI','DIH'});
        svr.UseInterleave=0;
    end 
    
    if isfield(svr,'DI')
        for v=1:svr.NV
            svr.VP{v}=repmat(dynInd(svr.VI{v},1,5),[ones(1,4) svr.paPerInte{v}(1)]);               
            for i=2:svr.I(v);svr.VP{v}=cat(5,svr.VP{v},repmat(dynInd(svr.VI{v},i,5),[ones(1,4) svr.paPerInte{v}(i)]));end
        end
        svr=rmfield(svr,{'DI','DIH'});
        svr.UseInterleave=0;
    end 
end

if svr.UseInterleave   
    DPI=cat(5,svr.VI{:});%Velocity
    PI=svr.I;
    MPI=svr.MInte;
else
    DPI=cat(5,svr.VP{:});%Velocity
    PI=svr.P;
    MPI=svr.MPack;
end
[gpu,tD,NT,dH,Tupd,E,NXX,spGridX,NXT,spGridT,ups,Fx,FHx,lambda,NPr,flagw,NcontTest,winit]=initializeDEstimation(DPI,svr.NX,svr.MSX,svr.Alg.NT,svr.Alg.Ups,svr.Alg.InvVar(2+typ),svr.Alg.NconvT,svr.Alg.Winit(2));

if ~isfield(svr,'convT');svr.convT=zeros([1 NT],'single');end
if ~isfield(svr,'w');svr.w=winit*ones([1 NT],'single');end
convT=svr.convT;w=svr.w;

indPack=mat2cell(1:sum(PI),1,PI);
parGroupingA=[];%To demean
parGroupingB=[];%To group convergence

fina=0;contIt=0;nIt=1;
while fina~=2 && nIt<svr.Alg.NItMaxD
    if nIt==2 && profi;profile on;end
    [Eprev,dH,convT,contIt,NcontTest,w]=initializeConvergenceControl(E,dH,contIt,NcontTest,nIt,debug,convT,w);
    
    cont=0;
    for v=1:svr.NV
        if any(convT(indPack{v})<svr.Alg.NconvT)
            [xT,W]=transformVolume;
            for p=1:PI(v)
                cV=cont+p;
                if convT(cV)<svr.Alg.NconvT
                    [Eprev,ry,xP,vt,v0]=computeCost(Eprev,DPI);                                      
                    ry=metricFiltering(ry,svr.G{2}{v});  
                    ry=metricMasking(ry,W,svr.Alg.RobustMotion);%For robust estimation
                    svr.yy{v}=ry;
                    svr=svrDecode(svr,2,[v p],2);
                    ry=svr.xx;
                    if gpu;ry=gpuArray(ry);end
                    
                    if ups(2)~=0;G=deformationGradientTensor(xP,spGridX,svr.GdX);
                    else G=deformationGradientTensorSpace(xP,spGridX);
                    end
                    dv=computeGradientHessianElastic(ry(:),G(:),NXX,NXT,svr.Ld,ups(2));
                    dv=integrateReducedAdjointJacobi(dv,vt,spGridT,svr.Ld,svr.GdT,ups(1));
                    dv=prod(spGridX)*(dv+lambda*v0/PI(v));
                    dH(:,cV)=dv(:);
                end
            end                        
            svr.y{v}=gather(svr.y{v});
        end
        cont=cont+PI(v);
    end
    dH=modulateGradient(NPr,dH,tD,DPI,svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]),svr.Alg.ThInvert(2+typ),convT<svr.Alg.NconvT);
    
    [flagw,fina,Einit,Tupr]=prepareLineSearch(flagw,nIt,Eprev,DPI);
    while fina==0         
        [Tup,Tupr,w,modw,indNotConv]=updateRule(NPr,dH,w,NXX.*spGridX/8,tD,svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]),convT,svr.Alg.NconvT,flagw,Tupr,DPI,parGroupingA,svr.Alg.ConstrainDiffeo,svr.Alg.ThInvert(2+typ));
        cont=0;
        for v=1:svr.NV
            if any(convT(indPack{v})<svr.Alg.NconvT) && any(flagw(indPack{v})~=2)
                [xT,W]=transformVolume;   
                for p=1:PI(v)
                    cV=cont+p;
                    if indNotConv(cV)
                        Tupd=dynInd(Tupd,cV,tD,dynInd(DPI,cV,tD)+dynInd(Tupr,cV,tD));
                        E=computeCost(E,Tupd);
                    end
                end
                svr.y{v}=gather(svr.y{v});
            end
            cont=cont+PI(v);
        end                            
        [flagw,w,Eprev,DPI,fina,convT,conv]=finalizeConvergenceControl(nIt,flagw,E,Eprev,Einit,convT,svr.Alg.NconvT,w,Tupd,DPI,fina,NXT,debug,parGroupingB,spGridX,svr.Alg.ConvL,svr.Alg.BaseConvLevel(2+typ),NT,modw,ups(1));
    end
    if nIt==2 && profi
        profile off;profsave(profile('info'),'/home/lcg13/Work/Profiling/ProfileB');
        1
        pause
    end
    nIt=nIt+1;
end
svr.w=w;svr.convT=convT;        
svr=rmfield(svr,{'w'});
if conv;svr=rmfield(svr,{'convT'});end

mergedVarA=cell(1,tD);  
for n=1:tD;mergedVarA{n}=size(DPI,n);end;mergedVarA{tD}=PI;
if svr.UseInterleave
    [svr.DI,svr.DIH,svr.DIJac,svr.DIRes,svr.VI]=computeDeformableTransforms(resPop(mat2cell(DPI,mergedVarA{:}),tD,[],2),svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]));
else
    [svr.DP,svr.DPH,svr.DPJac,svr.DPRes,svr.VP]=computeDeformableTransforms(resPop(mat2cell(DPI,mergedVarA{:}),tD,[],2),svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]));
end



function [Ein,ry,xP,vt,v0]=computeCost(Ein,Tin)
    xP=xT;
    v0=dynInd(Tin,cV,tD);    
    if nargout>=4;[phi,vt]=precomputeFactorsElasticTransform(v0,spGridT,svr.Alg.NG,svr.Ld,svr.GdT,ups(1));else [phi]=precomputeFactorsElasticTransform(v0,spGridT,svr.Alg.NG,svr.Ld,svr.GdT,ups(1));end
    xP=elasticTransform(xP,phi,spGridX,ups(3));    
    ry=encodingPropagation(xP,1);
    ry=metricMasking(ry,W,svr.Alg.RobustMotion);
    ry=metricFiltering(ry,svr.G{2}{v});            
    ry=metricMasking(ry,dynInd(MPI{v},p,5));
    Ein(cV)=prod(spGridX)*gather(ry(:)'*ry(:));    
    if lambda>0;Ein(cV)=Ein(cV)+lambda*prod(spGridT./(ups(1)+single(ups(1)==0)))*computeRiemannianMetric(v0,svr.Ld,spGridT,ups(1),[],Fx,FHx)/PI(v);end  
    
    function x=encodingPropagation(x,typEnc)
        svr.xx=x;            
        svr=svrEncode(svr,typEnc,[v p],2);
        x=svr.yy{v};
        if gpu;x=gpuArray(x);end        
    end   
end

function [xT,W]=transformVolume
    W=sqrt(svr.W{v});
    if gpu;svr.y{v}=gpuArray(svr.y{v});end
    xT=svr.x;
    Tf=precomputeFactorsSincRigidTransformQuick(svr.kGrid,svr.rGrid,dynInd(svr.TV,v,5),1,0,[],1,svr.cGrid);
    xT=real(sincRigidTransformQuick(xT,Tf,1,svr.FT));    
    if isfield(svr,'DV');xT=elasticTransform(xT,dynInd(svr.DV,v,5),svr.MSX,svr.Alg.Ups(3));end
end

end
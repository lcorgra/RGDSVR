function [svr,conv]=svrSolveDVolu(svr)

%SVRSOLVEDVOLU   Solve for a deformable transform D on a per-volume level
%   [SVR,CONV]=SVRSOLVEDVOLU(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure after an iteration of solver
%   ** CONV is a flag to indicate convergence
%

profi=0;%For profiling
debug=svr.Alg.Debug;%For seeing info

DV=svr.VV;%Velocity

[gpu,tD,NT,dH,Tupd,E,NXX,spGridX,NXT,spGridT,ups,Fx,FHx,lambda,NPr,flagw,NcontTest,winit]=initializeDEstimation(DV,svr.NX,svr.MSX,svr.Alg.NT,svr.Alg.Ups,svr.Alg.InvVar(1),svr.Alg.NconvT,svr.Alg.Winit(2));

if ~isfield(svr,'convT');svr.convT=zeros([1 NT],'single');end
if ~isfield(svr,'w');svr.w=winit*ones([1 NT],'single');end
convT=svr.convT;w=svr.w;

parGroupingA=[];%To demean
parGroupingB=[];%To group convergence

fina=0;contIt=0;nIt=1;
while fina~=2 && nIt<svr.Alg.NItMaxD
    if nIt==2 && profi;profile on;end
    [Eprev,dH,convT,contIt,NcontTest,w]=initializeConvergenceControl(E,dH,contIt,NcontTest,nIt,debug,convT,w);
    
    for v=1:svr.NV
        if convT(v)<svr.Alg.NconvT
            [xT,W]=transformVolume;
            [Eprev,ry,xP,vt,v0]=computeCost(Eprev,DV);
            ry=metricFiltering(ry,svr.G{2}{v});
            ry=metricMasking(ry,W,svr.Alg.RobustMotion);%For robust estimation
            svr.yy{v}=ry;
            svr=svrDecode(svr,2,v,3);
            ry=svr.xx;
            if gpu;ry=gpuArray(ry);end

            if ups(2)~=0;G=deformationGradientTensor(xP,spGridX,svr.GdX);
            else G=deformationGradientTensorSpace(xP,spGridX);
            end
            dv=computeGradientHessianElastic(ry(:),G(:),NXX,NXT,svr.Ld,ups(2));                
            dv=integrateReducedAdjointJacobi(dv,vt,spGridT,svr.Ld,svr.GdT,ups(1));
            dv=prod(spGridX)*(dv+lambda*v0);
            dH(:,v)=dv(:);
            svr.y{v}=gather(svr.y{v});
        end
    end
    if ups(1)==0;dH=real(dH);end    
    dH=modulateGradient(NPr,dH,tD,DV,svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]),svr.Alg.ThInvert(1),convT<svr.Alg.NconvT);          
    
    [flagw,fina,Einit,Tupr]=prepareLineSearch(flagw,nIt,Eprev,DV);
    while fina==0        
        [Tup,Tupr,w,modw,indNotConv]=updateRule(NPr,dH,w,NXX.*spGridX/8,tD,svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]),convT,svr.Alg.NconvT,flagw,Tupr,DV,parGroupingA,svr.Alg.ConstrainDiffeo,svr.Alg.ThInvert(1));     
        for v=1:svr.NV        
            if indNotConv(v)
                [xT,W]=transformVolume;
                Tupd=dynInd(Tupd,v,tD,dynInd(DV,v,tD)+dynInd(Tupr,v,tD));
                E=computeCost(E,Tupd);
                svr.y{v}=gather(svr.y{v});
            end
        end        
        [flagw,w,Eprev,DV,fina,convT,conv]=finalizeConvergenceControl(nIt,flagw,E,Eprev,Einit,convT,svr.Alg.NconvT,w,Tupd,DV,fina,NXT,debug,parGroupingB,spGridX,svr.Alg.ConvL,svr.Alg.BaseConvLevel(1),NT,modw,ups(1));
    end
    if nIt==2 && profi
        profile off
        profsave(profile('info'),'/home/lcg13/Work/Profiling/ProfileA')
        1
        pause
    end
    nIt=nIt+1;
end
svr.w=w;svr.convT=convT;        
svr=rmfield(svr,{'w'});
if conv;svr=rmfield(svr,{'convT'});end
[svr.DV,svr.DVH,svr.DVJac,svr.DVRes,svr.VV]=computeDeformableTransforms(DV,svr.Alg.NG,svr.Ld,svr.GdT,spGridT,ups([1 4]));


function [Ein,ry,xP,vt,v0]=computeCost(Ein,Tin)
    xP=xT;
    v0=dynInd(Tin,v,tD);
    if nargout>=4;[phi,vt]=precomputeFactorsElasticTransform(v0,spGridT,svr.Alg.NG,svr.Ld,svr.GdT,ups(1));else [phi]=precomputeFactorsElasticTransform(v0,spGridT,svr.Alg.NG,svr.Ld,svr.GdT,ups(1));end  
    xP=elasticTransform(xP,phi,spGridX,ups(3));
    ry=encodingPropagation(xP,1);      
    ry=metricMasking(ry,W,svr.Alg.RobustMotion);
    ry=metricFiltering(ry,svr.G{2}{v});
    Ein(v)=prod(spGridX)*gather(ry(:)'*ry(:));
    if lambda>0;Ein(v)=Ein(v)+lambda*prod(spGridT./(ups(1)+single(ups(1)==0)))*computeRiemannianMetric(v0,svr.Ld,spGridT,ups(1),[],Fx,FHx);end

    function x=encodingPropagation(x,typEnc)                  
        svr.xx=x;
        svr=svrEncode(svr,typEnc,v,3);
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
end

end
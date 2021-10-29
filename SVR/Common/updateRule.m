function [Tup,Tupr,w,modw,indNotConv]=updateRule(NPr,dH,w,lim,tD,NG,Ld,Gd,spGridT,ups,convT,NconvT,flagw,Tupr,DV,parGrouping,constrainDiffeo,thInvert)

%UPDATERULE   Reweights an elastic transform update obtained from a 
%gradient descent procedure so that it is constrained within certain bounds
%   [TUP,TUPR,W,INDNOTCONV]=UPDATERULE(NPR,DH,W,LIM,TD,NG,LD,GD,SPGRIDT,UPS,CONVT,NCONVT,FLAGW,TUPR,DV,PARGROUPING,CONSTRAINDIFFEO,NXX,NXT)
%   * NPR is the size of the elastic transform update
%   * DH is the gradient
%   * W is the weight
%   * LIM are the bounds
%   * TD is the dimension of different transforms
%   * NG is the number of temporal instants for geodesic shooting,
%   defaults to 10
%   * LD is a symmetric positive-definite differential operator
%   * GD is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * SPGRIDT is the spacing of the spatial grid
%   * UPS is a padding factor for circular convolutions (first component)
%   and for inverting the field (second component)
%   * CONVT indicates partial convergence status for different motion
%   states
%   * NCONVT is the required number of iterations with motion updates below
%   a certain threshold for establishing convergence
%   * FLAGW are flags used to indicate partial convergence for different
%   motion states
%   * TUPR is the update on the transform parameters
%   * DV are current transform parameters
%   * PARGROUPING indicates whether constrains should be established using
%   a group of transforms (if a vector) or globally (if empty)
%   * CONSTRAINDIFFEO serves to constrain to diffeomorphic deformations
%   * THINVERT is the threshold for invertibility
%   ** TUP is the constrained elastic transform update (Fourier domain)
%   ** TUPR is the update on the transform parameters
%   ** W is an updated weight
%   ** MODW indicates whether w has been modified
%   ** INDNOTCONV indicates not converged states for line search
%

if thInvert==0;thInvert=0.85;end%DefaultValue
indNotConv=convT<NconvT & flagw~=2;
diffTr=~indNotConv;
multA=2;%Factor to reduce the step length to ensure diffeomorphism
modw=w;modw(:)=0;

NV=NPr(tD);
while any(~diffTr)    
    %OBTAIN T UPDATE
    TuprFull=-bsxfun(@times,dH,1./w);
    TuprFull(:,w>=1e4)=0;
    TuprFull=reshape(TuprFull,NPr);

    %IF TOO BIG, THEN LIMIT
    maxT=zeros([ones(1,3) 3],'single');
    for v=1:NV
        if indNotConv(v)
            v0=dynInd(TuprFull,v,tD);
            phi=precomputeFactorsElasticTransform(v0,spGridT,NG,Ld,Gd,ups(1));
            maxT=max(maxT,gather(multDimMax(abs(phi),1:3)));
        end
    end
    maxT=maxT(:)';
    factW=max(maxT./lim);
    if factW>1
        w=w*factW;
        TuprFull=-bsxfun(@times,dH,1./w);
        TuprFull(:,w>=1e4)=0;
        TuprFull=reshape(TuprFull,NPr);
    end


    if ~isempty(parGrouping)        
        wrapMeanT=@(x)mean(x,tD);
        mergedVarA=cell(1,tD);
        mergedVarB=cell(1,tD);
        for n=1:tD;mergedVarA{n}=size(TuprFull,n);end;mergedVarA{tD}=parGrouping;
        [mergedVarB{1:tD-1}]=deal(1);mergedVarB{tD}=parGrouping; 
        TuprMean=cellfun(wrapMeanT,mat2cell(TuprFull,mergedVarA{:}),'UniformOutput',false);
        TuprFull=bsxfun(@minus,TuprFull,repelem(cat(tD,TuprMean{:}),mergedVarB{:}));
    end
    Tupr=dynInd(Tupr,indNotConv,tD,dynInd(TuprFull,indNotConv,tD));
    Tup=DV+Tupr;

    if constrainDiffeo
        for v=1:NV
            if ~diffTr(v)
                v0=dynInd(Tup,v,tD);
                phi=precomputeFactorsElasticTransform(v0,spGridT,NG,Ld,Gd,ups(1));
                [willConverge,~,~,~,Dgmax]=invertElasticTransform(phi,spGridT,ups(2),1,thInvert);                                
                if ~willConverge;w(v)=w(v)*multA;modw(v)=modw(v)+1;else diffTr(v)=true;end%%%modw helps to preserve the minimum energy in most cases, particularly if it takes the value 1, but it may not in exceptional circumnstances...
                if w(v)>=1e4
                    %fprintf('Maximum value reached: %d - %.3f\n',v,Dgmax);
                    w(v)=1e4;diffTr(v)=true;
                    Tupr=dynInd(Tupr,v,tD,0);Tup=dynInd(Tup,v,tD,dynInd(DV,v,tD));
                end
            end
        end
    else
        diffTr(:)=true;
    end
end

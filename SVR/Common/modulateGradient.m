function dH=modulateGradient(NPr,dH,tD,DV,NG,Ld,Gd,spGridT,ups,thInvert,flagUpdate)

%MODULATEGRADIENT   Modulates the gradient of a diffeomorphic field with local invertibility
%   DH=MODULATEGRADIENT(NPR,DH,TD,DV,LD,GD,SPGRIDT,UPS,THINVERT,FLAGUPDATE)
%   * NPR is the size of the elastic transform update
%   * DH is the gradient
%   * TD is the dimension of different transforms
%   * DV is the field
%   * NG is the number of temporal instants for geodesic shooting
%   * LD is a symmetric positive-definite differential operator
%   * GD is the gradient type, one of the following 
%   '1stFiniteDiscreteForward' / '1stFiniteDiscreteBackward' / 
%   '1stFiniteDiscreteCentered' (default) or if a cell, the filters along 
%   the different dimensions
%   * SPGRIDT is the spacing of the spatial grid
%   * UPS is a padding factor for circular convolutions (first component)
%   and for inverting the field (second component)
%   * THINVERT is the threshold for invertibility
%   * FLAGUPDATE is a flag indicating that a given volume is to be updated
%   ** DH is the modulated gradient
%

if thInvert==0
    return
else
    NV=NPr(tD);
    NDH=size(dH);
    dH=reshape(dH,NPr);
    for v=1:NV
        if flagUpdate(v)
            v0=dynInd(DV,v,tD);
            phi=precomputeFactorsElasticTransform(v0,spGridT,NG,Ld,Gd,ups(1));
            [~,~,Dgaux]=invertElasticTransform(phi,spGridT,ups(2),1);
            dHaux=dynInd(dH,v,5);
            if ups(1)~=0
                for m=1:3;dHaux=ifftGPU(dHaux,m);end
            end
            dHaux=bsxfun(@times,dHaux,abs(thInvert-Dgaux).^2);
            if ups(1)~=0
                for m=1:3;dHaux=fftGPU(dHaux,m);end
            end
            dH=dynInd(dH,v,5,dHaux);
        end
    end
    dH=reshape(dH,NDH);
end

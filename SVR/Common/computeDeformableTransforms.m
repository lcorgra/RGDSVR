function [D,DH,DJ,DR,V]=computeDeformableTransforms(V,NG,Ld,Gd,spGridT,ups)

%COMPUTEDEFORMABLETRANSFORMS   Computes the terms involved in applying a
%given deformable transform for reconstruction
%   [D,DH,DJ,DR,V]=computeDeformableTransforms(V,NG,LD,GDT,SPGRIDT,UPS,NXX,NXT)
%   * V is the velocity field (in Fourier domain)
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
%   ** D is the transform field
%   ** DJ is the Jacobian of the transform field
%   ** DH is the inverse transform field
%   ** DR are the residuals from inverting the field
%   ** V is the velocity field (in Fourier domain)
%

if iscell(V)
    NV=length(V);
    [D,DH,DJ,DR]=deal(cell(1,NV));    
    for v=1:NV
        Va=real(V{v});Vb=dynInd(Va,1,4);
        [D{v},DH{v},DJ{v},DR{v}]=deal(Va,Va,Vb,Vb);
        if any(V{v}(:)~=0)
            for p=1:size(V{v},5)
                [phi,~,J]=precomputeFactorsElasticTransform(dynInd(V{v},p,5),spGridT,NG,Ld,Gd,ups(1));
                D{v}=dynInd(D{v},p,5,phi);  
                DJ{v}=dynInd(DJ{v},p,5,J);       
                [phiinv,~,phires]=invertElasticTransform(phi,spGridT,ups(2));
                DH{v}=dynInd(DH{v},p,5,phiinv);
                DR{v}=dynInd(DR{v},p,5,phires);            
            end
        else
            DJ{v}(:)=1;
        end
    end     
else
    Va=real(V);Vb=dynInd(Va,1,4);
    [D,DH,DJ,DR]=deal(Va,Va,Vb,Vb);
    if any(V(:)~=0)
        for v=1:size(V,5)
            [phi,~,J]=precomputeFactorsElasticTransform(dynInd(V,v,5),spGridT,NG,Ld,Gd,ups(1));
            D=dynInd(D,v,5,phi);  
            DJ=dynInd(DJ,v,5,J);
            [phiinv,~,phires]=invertElasticTransform(phi,spGridT,ups(2));
            DH=dynInd(DH,v,5,phiinv);
            DR=dynInd(DR,v,5,phires);
        end
    else
        DJ(:)=1;
    end
end

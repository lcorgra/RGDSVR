function [dx,ddx]=computeGradientHessianElastic(r,J,NX,NT,L,ups)

%COMPUTEGRADIENTHESSIANELASTIC   Computes the gradient and the Hessian for
%elastic transforms. Note at the moment the Hessian is computed as
%suggested in M Hernandez, "Gauss–Newton inspired preconditioned
%optimization in large deformation diffeomorphic metric mapping," Phys Med
%Biol, 59:6085-6115, 2014
%   [DX,DDX]=COMPUTEGRADIENTHESSIANELASTIC(R,J,NX,NT,UPS)
%   * R are the residuals
%   * J is the Jacobian
%   * NX are the dimensions in image space
%   * NT are the dimensions in Fourier space
%   * {L} is a symmetric positive-definite differential operator
%   * {UPS} is a padding factor for circular convolutions
%   ** DX is the gradient
%   ** DDX is the Hessian
%

if isscalar(ups);ups=ups*ones(1,3);end

if ~any(ups==0)
    M=round(NX.*ups);ups=NX./M;
    if any(M~=NX)
        r=mapSpace(r,1,NX,M);J=mapSpace(J,1,NX,M);
        r=mapSpace(r,0,M,M);J=mapSpace(J,0,M,M);
    end
else
    M=NX;
end
r=reshape(r,[M 1 numel(r)/prod(M)]);J=reshape(J,[M 3 numel(r)/prod(M)]);
dx=-sum(real(bsxfun(@times,r,J)),5);
dx=mapSpace(dx,1,M,NT);
dx=bsxfun(@times,dx,1./L);
if any(ups==0);dx=mapSpace(dx,0,NT,NT);end

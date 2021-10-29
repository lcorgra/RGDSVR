function [E,dH,convT,contIt,NcontTest,w]=initializeConvergenceControl(E,dH,contIt,NcontTest,nIt,debug,convT,w)

%INICIALIZECONVERGENCECONTROL   Initializes variables related with 
%convergence control for elastic motion estimation in SVR
%   [E,DH,CONVT,CONTIT,NCONTEST,W]=INITIALIZECONVERGENCECONTROL(E,DH,CONTIT,NCONTEST,NIT,DEBUG,CONVT,W)
%   * E is current energy
%   * DH is the gradient update
%   * CONTIT is a counter of number of iterations for resetting convergence
%   * NCONTEST is the required number of iterations for resetting
%   convergence
%   * NIT is the total number of iterations
%   * DEBUG serves to show information
%   * CONVT indicates partial convergence status for different motion
%   states
%   * W are the weights in the update rule
%   ** E is the previous energy
%   ** DH is the gradient update
%   ** CONVT indicates partial convergence status for different motion
%   states
%   ** CONTIT is a counter of number of iterations for resetting
%   convergence
%   ** NCONTEST is the required number of iterations for resetting
%   convergen
%   ** W are the weights in the update rule
%

dH(:)=0;    
if mod(contIt,NcontTest)==0
    if debug;fprintf('Iteration %d Resetting motion states\n',nIt);end
    convT(:)=0;
    contIt=0;NcontTest=NcontTest+1;
end    
contIt=contIt+1;
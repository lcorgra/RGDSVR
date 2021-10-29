function [flagw,fina,E,T]=prepareLineSearch(flagw,nIt,E,T)

%PREPARELINESEARCH   Prepares structures for line search 
%   [FLAGW,FINA,E,T]=PREPARELINESEARCH(FLAGW,NIT,E,T)  
%   * FLAGW are flags used to indicate partial convergence for different
%   motion states
%   * NIT is the total number of iterations
%   * E is the energy
%   * T are the transform parameters
%   ** FLAGW are flags used to indicate partial convergence for different
%   motion states
%   ** FINA is a flag indicating line search status
%   ** E is the energy
%   ** T are the transform parameters
%

flagw(:)=1;    
if ismember(nIt,1);flagw(:)=0;end%We do line search
fina=0;
T(:)=0;

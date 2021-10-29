function svr=svrAlternateMinimization(svr)

%SVRALTERNATEMINIMIZATION   Performs an alternate minimization for SVR
%   SVR=SVRALTERNATEMINIMIZATION(SVR)
%   * SVR is a svr structure
%   ** SVR is a svr structure after reconstruction
%

profi=0;%For profiling

%COOLING SCHEDULE FOR ROBUST RECONSTRUCTION
coolSched=flip(linspace(svr.Alg.Lp,2,svr.Alg.NC));

%HARDCODED MAXIMUM NUMBER OF ITERATIONS FOR EACH MOTION ESTIMATE STEP
MaxL=5;%Maximum number of levels
maxItMot=2*svr.Alg.NC*ones(1,5);maxItMot(end+1:MaxL)=maxItMot(end);
nItMot=zeros(1,MaxL);
nItReset=ones(1,MaxL);
nItResetTotal=ones(1,MaxL);

conv=1;
motEst=svr.MotEst;
n=1;nF=1;
if ~svr.PreTr;svr.x(:)=0;end
non=2;

while(1)
    if svr.Alg.Debug && motEst<svr.Alg.Nl;fprintf('Outern iteration: %d / Exponent factor reconstruction: %.2f\n',nItMot(motEst+1)+1,non);end
    if n==1 && svr.Alg.Debug;tsta=tic;end
    if profi==1 && motEst==0
        save('/home/lcg13/Work/DataDeepFetal/svrProfiling.mat','svr','-v7.3');
        profile on;
    end
    %SOLVE CG
    svr=svrCG(svr,3+(nF<2)+(nF<3));            
    
    %SOLVE DD
    if isempty(svr.F);[svr.dd.net,svr.xz]=svrDD(svr,1);end
    
    %CONVERGENCE OF RECONSTRUCTION
    if motEst>=svr.Alg.Nl || n>svr.Alg.NItMax;break;end        
    
    %MOTION AND NORMALIZATION ESTIMATION
    if nItMot(motEst+1)<maxItMot(motEst+1)
        if motEst==0
            conv=1;
            if non~=svr.Alg.Lp;conv=0;end
        elseif motEst==1
            [svr,conv]=svrSolveTVolu(svr);
            if non~=svr.Alg.Lp;conv=0;end 
        elseif motEst==2
            [svr,conv]=svrSolveDVolu(svr);      
            if non~=svr.Alg.Lp;conv=0;end           
        elseif motEst==3        
            [svr,conv]=svrSolveDPack(svr,0);      
            if non~=svr.Alg.Lp;conv=0;end
        elseif motEst==4
            [svr,conv]=svrSolveDPack(svr,1);
            if non~=svr.Alg.Lp;conv=0;end
        end
    else
        fprintf('Motion correction at level %d terminated without reaching convergence\n',motEst);
        conv=1;svr.w=[];svr.convT=[];svr=rmfield(svr,{'w','convT'});
    end
    
    %CONVERGENCE OF ESTIMATION
    if motEst<=MaxL-1
        m=motEst+1;
        nItMot(m)=nItMot(m)+1;
        motEst=motEst+conv;        
        if nItMot(m)==nItResetTotal(m) || ismember(motEst,2:MaxL)
            if isfield(svr,'convT');svr=rmfield(svr,'convT');end
            nItReset(m)=nItReset(m)+1;
            nItResetTotal(m)=nItResetTotal(m)+nItReset(m);            
        end
    end
    if conv && svr.Alg.Debug;tend=toc(tsta);fprintf('Time level %d: %.3f s\n',motEst-1,tend);end
  
    if conv==1 && motEst<svr.Alg.Nl 
        n=1;non=2;
    else              
        if ~conv;non=coolSched(min(nItMot(m)+1,svr.Alg.NC));
        else non=svr.Alg.Lp;
        end
        n=n+1;
    end
    %ROBUSTNESS
    svr=svrSliceWeights(svr,non);    
    nF=nF+1;

    if profi && motEst==0 && n~=1
        profile off;profsave(profile('info'),'/home/lcg13/Work/Profiling/ProfileB');
        1
        pause
    end    
end
if n>svr.Alg.NItMax;fprintf('Alternate minimization solver terminated without reaching convergence\n');end 

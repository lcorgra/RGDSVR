function [K,R]=channelsDeepDecoder(L,N,M,Rtarget,parNet)

%CHANNELSDEEPDECODER   Computes the number of channels of the DD for a
%target compression ratio
%   [K,R]=CHANNELSDEEPDECODER(L,N,M,RTARGET,PARNET)
%   * L is the number of levels of the network
%   * N is the size of the target image
%   * M is the number of channels of the target image
%   * RTARGET is the target compression ratio
%   * PARNET is a structure with network parameters
%   ** K is the number of channels in the finest resolution level
%   ** R is the actual compression ratio
%


NM=prod(N)*M;
K0=ceil(sqrt(NM/(Rtarget*max(L-2,1))));%Upper bound
R=0;
K0=K0+1;
while R<Rtarget
    K0=K0-1;
    K=[ceil(K0*ones(1,L+2).*(parNet.fact_k.^(L+1:-1:0))) M];%Filter plan
    NP=sum(K(1:L+2).*K(2:L+3).*(parNet.NCo.^parNet.ND));
    if parNet.use_bias;NP=NP+sum(K(2:L+3));end
    if parNet.use_conv_bn~=-1;NP=NP+2*sum(K(2:L+2));end%Number of parameters of deep decoder
    if parNet.use_act==4
        NP=NP+sum(K(2:L+2).*ceil(K(2:L+2)/parNet.Kred));%Convolution
        NP=NP+2*(sum(K(2:L+2))+sum(ceil(K(2:L+2)/parNet.Kred)));%Batch normalization
        if parNet.use_bias_atac;NP=NP+sum(K(2:L+2))+sum(ceil(K(2:L+2)/parNet.Kred));end%Bias
    end  
    NP=NP*2^(parNet.wav_L*parNet.ND);
    R=NM/NP;%Ratio of compression
end

function [L,rel_err,base,Q,iter]=lowrank_corr(X,r,sigma,epsilon,q)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        GoDec+ Algotithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%The code is revised from the code of GoDec provided by Tianyi Zhou
%INPUTS:
%X: nxp data matrix with n samples and p features
%rank: rank(L)<=rank
%q: >=0, power scheme modification, increasing it lead to better
%OUTPUTS:
%L:Low-rank part
%RMSE: Relative error
%Q: Basis of the low-rank data L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%REFERENCE:
%Kailing Guo, Liu Liu, Xiangmin Xu, Dong Xu, and Dacheng Tao, "GoDec+: Fast and Robust Low-rank Matrix
%Decomposition Based on Maximum Correntropy", TNNLS 2017
%Author: Kailing Guo


%iteration parameters
% rng('default')
iter_max=1e+2;


rel_err=[];

X = X';

[m,n]=size(X);


T = zeros(size(X));
L = X;

    iter = 1;
    Y2=randn(n,r);

    while true
        e = T - T.*exp(-T.*T/sigma);
        X1=X-e;

            %Update of L
            for i=1:q+1
                Y1=X1*Y2;
                Y2=X1'*Y1;
            end
            [Q,R]=qr(Y2,0);
            base = X1*Q;
            L_new=base*Q';
            Y2 = Q;

        T = X - L_new;

        L_diff = L_new - L;
        stop_cri = (norm(L_diff(:))/norm(L(:)))^2;
        rel_err = [rel_err,stop_cri];

        if stop_cri < epsilon || iter >iter_max
            break;
        end

        L = L_new;
        iter = iter + 1;
    end
    L = L_new;
    L = L';

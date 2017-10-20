%Applying GoDec+ for subspace clustering
clear
load('yaleB_cropped.mat')

k = 5;
testSet = dataset(1:k);

labels = [];
for i = 1:k
    labels = [labels;ones(size(testSet{i},2),1)*i];
end
testSet = cell2mat(testSet);

testSet_tmp = zeros(48*42,size(testSet,2));
for i = 1:size(testSet,2)
    temp = reshape(testSet(:,i),[192,168]);
    temp = imresize(temp,[48,42]);
    testSet_tmp(:,i) = temp(:);
end
testSet = testSet_tmp;

q = 0;

Sigma = 360;

R = round(min(size(testSet))*0.8);

Lambda = 9e+5;

epsion = 1e-7;
tic
Acc = zeros(length(R),length(Sigma),length(Lambda));
Err = zeros(length(R),length(Sigma),length(Lambda));
for r_i = 1:length(R)
    r = R(r_i);
    
    Acc_tmp = zeros(1,length(Sigma),length(Lambda));
    Err_tmp = zeros(1,length(Sigma),length(Lambda));
    for t_i = 1:length(Sigma)
        sigma = Sigma(t_i);
        Acc_row = zeros(1,length(Lambda));
        Err_row = zeros(1,length(Lambda));
        [L,RMSE,base,Q]=lowrank_corr(testSet,r,sigma,epsion,q);

        LtL = L'*L;

        for l = 1:length(Lambda)
            lambda = Lambda(l);

                iter = 1;
                e = zeros(size(testSet));
                
                H0 = zeros(size(testSet,2));
                testSet_sq = testSet'*testSet;
                P = (LtL+lambda*eye(size(LtL)))\L';
                while true

                    testSet_tmp = L - e;
                    H = P*testSet_tmp;
                    T = L - L*H;
                    T_sq =T.*T;

                    e = T - T.*exp(-T.*T/sigma);
                    tmp = H-H0;
                    if norm(tmp(:))<1e-10 || iter >100
                        break;
                    end
                    H0 = H;
                    iter = iter + 1;
                end
                [U,D,V]=svd(H,0);
                
                r1 = r;
                U = U(:,1:r1);
                D = sqrt(D(1:r1,1:r1));
                U = U*D;
                
                for i = 1:size(U,1)
                    U(i,:) = U(i,:)/norm(U(i,:));
                end
                A = U*U';
                A = A.^4;
                grps = SpectralClustering(A,k);
                missrate = 1-compute_AC(labels,grps);
                

            Err_row(l) = mean(missrate);
        end

        Err_tmp(1,t_i,:) = Err_row;
    end
    Err(r_i,:,:) = Err_tmp;
end
toc

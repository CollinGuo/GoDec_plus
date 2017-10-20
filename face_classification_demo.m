%Applying GoDec+ for face classification
clear
load('CMU_PIE.mat')
for i = 1:length(dataset)
    dataset{i} = dataset{i}(:,1:42);
    dataLabel{i} = datalabel{i}(1:42);
end

r=1:6;
rep_times = 10;

c_rate = 0.01;
q = 0;

sigma = 1e-3;

epsilon = 1e-7;
dim = r*length(dataset);

accuracy = zeros(length(r),rep_times);

for r_i = 1:length(r)
    Re = zeros(1,rep_times);
    parfor k = 1:rep_times

        trainSet = cell(1,length(dataset));
        testSet = cell(1,length(dataset));
        trainLabel = cell(length(dataset),1);
        testLabel = cell(length(dataset),1);
        for i = 1:length(dataset)
            idx = randperm(size(dataset{i},2));
            trainSet{i} = dataset{i}(:,idx(1:7));
            testSet{i} = dataset{i}(:,idx(8:end));
            trainLabel{i} = datalabel{i}(idx(1:7));
            testLabel{i} = datalabel{i}(idx(8:end));
        end
        
        trainSet0 = trainSet;
        trainSet = cell2mat(trainSet);
        testSet = cell2mat(testSet);
        trainLabel = cell2mat(trainLabel);
        testLabel = cell2mat(testLabel);
        card = round(numel(trainSet)*c_rate);
        for i = 1:size(trainSet,2)
            trainSet(:,i) = trainSet(:,i)/norm(trainSet(:,i));
        end
        for i = 1:size(testSet,2)
            testSet(:,i) = testSet(:,i)/norm(testSet(:,i));
        end

                    L = cell(1,length(dataset));
                    id = 0;
                    for i = 1:length(L)
                        trainSet0{i} = trainSet(:,id+1:id+size(trainSet0{i},2));
                        id = id +size(trainSet0{i},2);
                        [L{i},RMSE,~,Q]=lowrank_corr(trainSet0{i},r(r_i),sigma,epsilon,q);
                        [L{i},~]=qr(L{i},0);
                        L{i} = L{i}(:,1:r(r_i));
                    end
                    L = cell2mat(L);
                    H0 = zeros(size(L,2),size(testSet,2));
                    iter = 1;
                    e = zeros(size(testSet));
                    while true
                        testSet_tmp = testSet-e;
                        H = pinv(L)*testSet_tmp;
                        T = testSet - L*H;
                        T_sq = T.*T;
                        e = T - T.*exp(-T_sq/sigma);
                        tmp = H-H0;
                        if norm(tmp(:))<1e-7 || iter >100
                            break;
                        end
                        H0 = H;
                        iter = iter + 1;
                    end
                    corr = zeros(length(dataset),size(testSet,2));
                    r_ = r(r_i);
                    for i = 1:length(dataset)
                        tmp = testSet - L(:,((i-1)*r_+1):i*r_)*H(((i-1)*r_+1):i*r_,:);
                        corr(i,:) = sum(exp(-tmp.*tmp/sigma),1);
                    end
                    [~,result_label]=max(corr,[],1);
                    result_label = result_label';

            result = sum(result_label==testLabel)/length(testLabel);

        Re(k) = result;
    end
    accuracy(r_i,:) = Re;

end

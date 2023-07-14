% The code is written by Jerry. For any problems, please contact jiewen_pr@126.com.
% The code is for the paper
% Wai Keung Wong, Jie Wen, et al., 
% Neighbor group structure preserving based consensus graph learning for incomplete multi-view clustering, 
% in information fusion, 2023.
% if you use the code, please cite the above reference above.

clear;
clc

Dataname = 'bbcsport4vbigRnSp';

percentDel = [0.5]
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];

rand('seed',8586);
lambda1 = 0.1;
lambda2 = 0.00001;
lambda3 = 0.0001;
para_k = 5;
f = 3;
para_r = 8;

load(Dataname);
load(Datafold);
ind_folds = folds{f};
truthF = truth;  
clear truth
numClust = length(unique(truthF));
num_view = length(X);
NumSamp = length(truthF);
if size(X{1},2)~=NumSamp
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end

linshi_WW = 0;
for iv = 1:num_view
    X1 = X{iv};
    X1 = NormalizeFea(X1,0);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(:,ind_0) = 0;
    X{iv} = X1;
    X1(:,ind_0) = [];  
    Y{iv} = X1;            
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;
    linshi_W = ones(NumSamp,NumSamp);
    linshi_W(ind_0,:) = 0;
    linshi_W(:,ind_0) = 0;
    Wiv{iv} = linshi_W;
    clear ind_0
    linshi_WW = linshi_WW+Wiv{iv};
end
clear X1 W1

for iv = 1:num_view
    options = [];
    options.NeighborMode = 'KNN';
    options.k = para_k;
% options.k = 0;
    options.WeightMode = 'HeatKernel';  % HeatKernel  Binary
    Z1 = full(constructW(Y{iv}',options));
    Sk_ini{iv} = G{iv}'*Z1*G{iv};

    options.WeightMode = 'Binary';  % HeatKernel  Binary
    Z1 = full(constructW(Y{iv}',options));
    Z1 = Z1+eye(size(Z1));
    Sb_ini{iv} = G{iv}'*Z1*G{iv};    
    clear Z1
end

F_ini = solveF(Sk_ini,numClust);

max_iter = 100;

[F,Z,obj] = NNGSI_CGL(X,Sk_ini,Sb_ini,Wiv,F_ini,numClust,lambda1,lambda2,lambda3,para_r,max_iter);
new_F = F;
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 

pre_labels    = kmeans(real(new_F),numClust,'emptyaction','singleton','replicates',20,'display','off');
result_cluster = ClusteringMeasure(truthF, pre_labels)*100  

                          



function [F,Z,obj] = NNGSI_CGL(X,Skiv,Sbiv,Wiv,F_ini,num_clust,lambda1,lambda2,lambda3,para_r,max_iter)

num_view = length(Skiv);

F = F_ini;

for iv = 1:num_view
    linshiXZ = X{iv}*Sbiv{iv}*diag(1./sum(Sbiv{iv},1));%用.*效率更高？为什么会出现NaN?因为有缺失，有0
    linshi_H = EuDist2(linshiXZ',linshiXZ',0);
    linshi_H = linshi_H.*Wiv{iv};%W乘在了这里
    linshi_H = linshi_H - diag(diag(linshi_H));%对角线置0
    linshi_H(isnan(linshi_H)) = 0;
    Eiv{iv} = linshi_H;
end
alpha = ones(1,num_view);
alpha_r = alpha.^para_r;
for iter = 1:max_iter

    % ------------ Z -------------- %
    linshi_fenzi = 0;
    linshi_fenmu = 0;
    for iv = 1:num_view
        linshi_fenzi = linshi_fenzi + alpha_r(iv)*(Skiv{iv}.*Wiv{iv}-0.5*lambda1*Eiv{iv});
        linshi_fenmu = linshi_fenmu + alpha_r(iv)*Wiv{iv};
    end
    linshi_P = EuDist2(F,F,0);
    linshi_P = linshi_P - diag(diag(linshi_P));
    linshi_Z = (linshi_fenzi-0.25*lambda3*linshi_P)./(linshi_fenmu+lambda2);
    Z = zeros(size(linshi_Z));
    for in = 1:size(Z,2)
        linshi_c = 1:size(linshi_Z,1);
        linshi_c(in) = [];
        Z(linshi_c,in) = EProjSimplex_new(linshi_Z(linshi_c,in));
    end
    % ----------- F ------ %
    linshiZ = (Z+Z')*0.5;
    LapZ = diag(sum(linshiZ))-linshiZ;
    [F, ~,~] = eig1(LapZ, num_clust, 0);
    % ------- alpha ------- %
    for iv = 1:num_view
        Rec_error(iv) = norm((Z-Skiv{iv}).*Wiv{iv},'fro')^2+lambda1*sum(sum(Eiv{iv}.*Z));%少乘了一个W
    end
    linshi_H = bsxfun(@power,Rec_error, 1/(1-para_r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,linshi_H,sum(linshi_H));         % alpha = H./sum(H);
    alpha_r = alpha.^para_r;
    % ----------------- obj ----------- %
    linshiZ = (Z+Z')*0.5;
    LapZ = diag(sum(linshiZ))-linshiZ;
    obj(iter) = alpha_r*Rec_error'+lambda3*trace(F'*LapZ*F)+lambda2*norm(Z,'fro')^2;
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-5
        iter
        break;
    end
end

end
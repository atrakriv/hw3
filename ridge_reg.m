
%%% 1 Implementing the ridge regression solver
clc; clear; close all;
load('diabetes.mat')
N_train = size(x_train,1);
N_test = size(x_test,1);
landas = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10];
bias_train = ones(N_train,1);
bias_test = ones(N_test,1);
x = [x_train bias_train];
x_test = [x_test bias_test];
t = y_train;
t_test = y_test;
k = 5;

%%%% 2 Training regression models on the Diabetes dataset
for i = 1 : length(landas)
    w = inv(landas(i)*eye(size(x_train,2)+1) + x'*x)*x'*t;
    %%% train error
    y = x*w;
    e = y - t;
    MSE_train(i) = norm(e)/N_train;
    %%% test error
    y_test = x_test*w;
    e_test = y_test - t_test;
    MSE_test(i) = norm(e_test)/N_test;
end
plot(log10(landas),MSE_train,'g','LineWidth',2);
hold on
plot(log10(landas),MSE_test,'b','LineWidth',2);
xlabel('log(\lambda)')
ylabel('MSE')
legend('training set','test set')

%%%% 3 Training regression models on the Diabetes dataset
out = floor(N_train/k);
in = N_train - out;
MSE_validation = zeros(length(landas),1);
for j = 1 : length(landas)
    for i = 1:k
        x_fold = x;
        x_fold((i-1)*out+1:i*out,:)=[];
        x_validation = x((i-1)*out+1:i*out,:);
        
        t_fold = t;
        t_fold((i-1)*out+1:i*out,:)=[];
        t_validation = t((i-1)*out+1:i*out,:);
              
        w_fold = inv(landas(j)*eye(size(x_train,2)+1) + x_fold'*x_fold)*x_fold'*t_fold;
        
        %%% validation set
        y_validation = x_validation*w_fold;
        e_validation = y_validation - t_validation;
        MSE_validation(j) = MSE_validation(j)+ norm(e_validation)/out;
    end
    MSE_validation(j) = MSE_validation(j)/k;
end

figure(2)
plot(log10(landas),MSE_validation,'b','LineWidth',2)
xlabel('log(\lambda)')
ylabel('MSE')
title('5-fold cross-validation')

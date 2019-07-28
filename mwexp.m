%addpath(genpath('F:\'))

clear all
% Example Usage
input=rand(3,10);
target=[1 ;2 ; 3];
net=cdRVFLtrain(input, target, 3, [10,5]);
y=cdRVFLtest(input, net)
% check target and y values


%iris
clear all
load fisheriris.mat
trainlabel=[ones(1,40), 2*ones(1,40), 3*ones(1,40)]';
traindata=[meas(1:40,:); meas(51:90,:); meas(101:140,:)];

testlabel=[ones(1,10), 2*ones(1,10), 3*ones(1,10)]';
testdata=[meas(41:50,:); meas(91:100,:); meas(141:150,:)];

tic, net=cdRVFLtrain(traindata, trainlabel, 1, [20 10]);toc
y=cdRVFLtest(testdata, net);


%mnist
clear all
load mnist.mat
trainX=double(trainX);trainX=trainX(1:10000,:);
trainY=double(trainY);trainY=trainY(1:10000);

testX=double(testX);testX=testX(1:200,:);
testY=double(testY);testY=testY(1:200);

tic, net=cdRVFLtrain(trainX, trainY', 10, [300, 15]);toc
y=cdRVFLtest(testX,net);








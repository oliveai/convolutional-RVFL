%experiment 1
clearvars,
input=rand(18,50);
target=[ones(1,6), ones(1,6)*2, ones(1,6)*3]';
net=cdRVFLtrain(input, target, 5, [8,3]);
out=cdRVFLtest(input, net) % check target and y values

%experiment 2
%iris data
clearvars,
load fisheriris.mat
trainlabel=[ones(1,40), 2*ones(1,40), 3*ones(1,40)]';
traindata=[meas(1:40,:); meas(51:90,:); meas(101:140,:)];

testlabel=[ones(1,10), 2*ones(1,10), 3*ones(1,10)]';
testdata=[meas(41:50,:); meas(91:100,:); meas(141:150,:)];

tic, net=cdRVFLtrain(traindata, trainlabel, 1, [50, 10]);toc
out=cdRVFLtest(testdata, net);


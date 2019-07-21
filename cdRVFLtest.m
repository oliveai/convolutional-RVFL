function y=cdRVFLtest(input,net)
% cdRVFLtest: ConvNET Random Vector Functional Link testing function
%
%Output Parameters
%         y: actul output
%
%Input Parameters
%         net: structure that includes network parameters.
%         fcweights, stride, convkernelflipped, poolkernel
%         outputlayerweights, fclayerstructure, numberofconvlayer
%
% Example Usage
%         input=rand(3,25);
%         target=rand(3,1);
%         net=cdRVFLtrain(input, target, 5, [8,3])
%         y=cdRVFLtest(input, net)
%        % check target and y values
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           TEST                               %
% %           ConvNET Random Vector Functional Link              %
% %                       (Avaraging)                            %
% %                  Apdullah Yayik, 2019                        %
% %                  apdullahyayik@gmail.com                     %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

convlayerouts=input;
for p=1:net.numberofconvlayer-1
    convlayerouts=conv2(convlayerouts, net.conv.kernelflipped, 'same'); % convolution
    convlayerouts=trans(batchN(convlayerouts), 'ReLU'); % Batch Normalization & ReLU
    temp = conv2(convlayerouts, net.pool.kernel, 'valid'); % pooling
    convlayerouts = temp(1:net.pool.stride:end, 1:net.pool.stride:end);
end

fclayerouts1=batchN(convlayerouts*net.fcweights{1,1}); % FC, Batch Normalization & softmax
fclayerouts1(:,net.dropoutlayers)=[]; % dropout layer
fclayerouts2=trans(batchN(fclayerouts1*net.fcweights{1,2}), 'softmax'); % FC, Batch Normalization & softmax
D=[input, fclayerouts1,fclayerouts2];
y=D*net.outputlayerweights; 
end















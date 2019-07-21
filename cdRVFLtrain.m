function net= cdRVFLtrain(input, target, numberofconvlayer, fclayerstructure)
% cdRVFLtrain: ConvNET Random Vector Functional Link training function
% 0.5 probality rate of dropout
%Output Parameters
%         net: structure that includes network parameters.
%         fcweights, stride, convkernelflipped, poolkernel
%         outputlayerweights, fclayerstructure, numberofconvlayer
%
%Input Parameters
%         input: input data (each row represent different observations)
%         target: desired outputs
%         numberofconvlayer: conv layer neuron numbers
%         fclayerstructure: neuron numbers of fully connected layers, for instance [5 8]
%
% Example Usage
%         input=rand(3,25);
%         target=rand(3,1);
%         net=cdRVFLtrain(input, target, 5, [8,3])
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                           TRAIN                              %
% %            ConvNET Random Vector Functional Link             %
% %                                                              %
% %                    Apdullah Yayik, 2019                      %
% %                    apdullahyayik@gmail.com                   %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isequal(size(target,1), size(input,1))==0
    error('Error: input and target sizes dismatch')
else
    net.numberofconvlayer=numberofconvlayer;
    convlayerouts=input;
    net.conv.kernel=rand(1,3); net.conv.kernelflipped=rot90(net.conv.kernel,2);
    net.pool.kernel=ones(1,2)/2; net.pool.stride=1;
    net.dropoutlayers=randperm(fclayerstructure(1),fclayerstructure(1)*0.5);
    
    for p=1:net.numberofconvlayer-1
        convlayerouts=conv2(convlayerouts, net.conv.kernelflipped, 'same');% convolution
        convlayerouts=trans(batchN(convlayerouts), 'ReLU'); % Batch Normalization & ReLU
        temp = conv2(convlayerouts, net.pool.kernel, 'valid'); % pooling
        convlayerouts = temp(1:net.pool.stride:end, 1:net.pool.stride:end);
    end
    
    net.fcweights{1,1}=rand( size(convlayerouts, 2),fclayerstructure(1));
    fclayerouts1=batchN(convlayerouts*net.fcweights{1,1}); % FC, Batch Normalization & softmax
    fclayerouts1(:,net.dropoutlayers)=[];% dropout layer
    net.fcweights{1,2}=rand( size(fclayerouts1, 2),fclayerstructure(2));
    fclayerouts2=trans(batchN(fclayerouts1*net.fcweights{1,2}), 'ReLU'); % FC, Batch Normalization & softmax
    D=[input, fclayerouts1, fclayerouts2];
    net.outputlayerweights=pinv(D)*target; % Pseudoinverse learning, svd
end







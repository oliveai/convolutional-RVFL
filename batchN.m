function n=batchN(H)
% batchN: Batch normalization
% Toffe and Szegedy, 2015
% recomended before non-linear transfering
%             
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                                                              %
% %                  Batch Normalization                         %
% %                                                              %
% %                  Apdullah Yayik, 2019                        %
% %                  apdullahyayik@gmail.com                     %
% %                                                              %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n=(H-mean(H,2))./std(H,0,2);
end
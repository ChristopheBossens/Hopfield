%% A memory which forgets
clc;clear all;
cd 'C:\Users\Christophe\Documents\My Dropbox\Code\Neuronal dynamics\Chapter 17. Memory and attractor dynamics'
hopfield = Hopfield;
%% Network capacity
% Generate random patterns
nPatterns = 50;
networkSize = 100;
nRepetitions = 10;
patternMatrix = rand(nPatterns,networkSize);
patternMatrix(patternMatrix(:) > 0.5) = 1;
patternMatrix(patternMatrix(:) < 0.9) = -1;

%%
hopfield.ResetWeights();
weightMatrixStatistics = zeros(2,nPatterns);
itMatrix = -1.*ones(nPatterns,nPatterns);
cMatrix = -1.*ones(nPatterns, nPatterns);
for patternIndex = 1:nPatterns
   hopfield.AddPattern((networkSize^(-0.5)).*patternMatrix(patternIndex,:));
   weightMatrix = hopfield.GetSynapseWeights();
   weightMatrixStatistics(1,patternIndex) = min(weightMatrix(:));
   weightMatrixStatistics(2,patternIndex) = max(weightMatrix(:));
   
   itData = zeros(1,nRepetitions);
   cData =  zeros(1,nRepetitions);
   
   for testPatternIdx = 1:patternIndex
       for repIdx = 1:nRepetitions
           netInput = hopfield.DistortPattern(patternMatrix(testPatternIdx,:),0.10);
           
           converged = -1;
           nIterations = 1;
           while converged == -1
               netOutput = hopfield.Iterate(netInput, 'async');

               if (sum(netInput-netOutput) == 0)
                   converged = 1;
               else
                   nIterations = nIterations + 1;
                   netInput = netOutput;
               end
           end

           itData(repIdx) = nIterations;
           cData(repIdx) = (netOutput*patternMatrix(testPatternIdx,:)')/networkSize;
       end
       itMatrix(patternIndex,testPatternIdx) = mean(itData);
       cMatrix(patternIndex,testPatternIdx) = mean(cData);
   end
end

clf
subplot(1,2,1),imshow(itMatrix',[]),colormap jet, colorbar,title('#Iterations'),ylabel('Test pattern'),xlabel('#Trained patterns')
subplot(1,2,2),imshow(cMatrix', []),colormap jet, colorbar,title('Pattern overlap')
%% Network capacity with thresholded synaptic weights
hopfield.ResetWeights();
weightMatrixStatistics2 = zeros(2,nPatterns);
itMatrix2 = -1.*ones(nPatterns,nPatterns);
cMatrix2 = -1.*ones(nPatterns, nPatterns);
for patternIndex = 1:nPatterns
   hopfield.AddPattern((networkSize^(-0.25)).*patternMatrix(patternIndex,:),0,0,'self',0.7);
   weightMatrix = hopfield.GetSynapseWeights();
   weightMatrixStatistics2(1,patternIndex) = min(weightMatrix(:));
   weightMatrixStatistics2(2,patternIndex) = max(weightMatrix(:));
   
   itData = zeros(1,nRepetitions);
   cData =  zeros(1,nRepetitions);
   
   for testPatternIdx = 1:patternIndex
       for repIdx = 1:nRepetitions
           netInput = hopfield.DistortPattern(patternMatrix(testPatternIdx,:),0.10);

           converged = -1;
           nIterations = 1;
           while converged == -1
               netOutput = hopfield.Iterate(netInput, 'async');

               if (sum(netInput-netOutput) == 0)
                   converged = 1;
               else
                   nIterations = nIterations + 1;
                   netInput = netOutput;
               end
           end

           itData(repIdx) = nIterations;
           cData(repIdx) = (netOutput*patternMatrix(testPatternIdx,:)')/networkSize;
       end
       itMatrix2(patternIndex,testPatternIdx) = mean(itData);
       cMatrix2(patternIndex,testPatternIdx) = mean(cData);
   end
end
%%
clf
subplot(2,3,1),imshow(itMatrix',[]),colormap jet,colorbar,title('#Iterations'),ylabel('Test pattern'),xlabel('#Trained patterns')
subplot(2,3,2),imshow(cMatrix', []),colormap jet,colorbar,title('Pattern overlap')
subplot(2,3,3),plot(1:nPatterns,weightMatrixStatistics(1,:),'r','LineWidth',2),hold on,
plot(1:nPatterns,weightMatrixStatistics(2,:),'g','LineWidth',2)

subplot(2,3,4),imshow(itMatrix2',[]),colormap jet,colorbar,title('#Iterations'),ylabel('Test pattern'),xlabel('#Trained patterns')
subplot(2,3,5),imshow(cMatrix2', []),colormap jet,colorbar,title('Pattern overlap')
subplot(2,3,6),plot(1:nPatterns,weightMatrixStatistics2(1,:),'r','LineWidth',2),hold on,
plot(1:nPatterns,weightMatrixStatistics2(2,:),'g','LineWidth',2)

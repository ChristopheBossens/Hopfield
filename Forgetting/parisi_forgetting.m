%% The following code implements the method discussed in:
% Parisi, G. (1986). A memory which forgets. Journal of Physics A:
% Mathematical and General
clc;clear;
hopfield = Hopfield;
nPatterns = 32;
networkSize = 64;

patternMatrix = zeros(nPatterns,networkSize);
for patternIdx = 1:nPatterns
    patternMatrix(patternIdx,:) = hopfield.GeneratePattern(networkSize,0.5);
end
%% The network is trained on all the patterns, by adding an additional parameter to
% the AddPattern function of the hopfield model, we can constrain the
% weightmatrix so that weights can no longer grow without bounds

hopfield.ResetWeights();
iterationMatrix = zeros(nPatterns);
overlapMatrix = zeros(nPatterns);
weightValues = zeros(3,nPatterns);
proportionRecalled = zeros(1,nPatterns);

% Set the simulation patterns
recallCriterion = 0.9;
nRepetitions = 5;
noiseLevel = .0;

for trainingPatternIdx = 1:nPatterns
    % Add pattern and inspect statistics of weight matrix
   hopfield.AddPattern(patternMatrix(trainingPatternIdx,:),1/(networkSize),0.03);
   
   currentWeights = hopfield.GetWeightMatrix();
   weightValues(1,trainingPatternIdx) = max(currentWeights(:));
   weightValues(2,trainingPatternIdx) = mean(currentWeights(:));
   weightValues(3,trainingPatternIdx) = min(currentWeights(:));
    
   % Test recollection on all previously learned patterns
   iterationVector = zeros(1,nRepetitions);
   overlapVector =  zeros(1,nRepetitions);
   
   for testPatternIdx = 1:trainingPatternIdx
       for repetitionIdx = 1:nRepetitions
           netInput = hopfield.DistortPattern(patternMatrix(testPatternIdx,:),noiseLevel);
           
           converged = -1;
           iterations = 1;
           while converged == -1
               netOutput = hopfield.Iterate(netInput, 'async');

               if (sum(netInput-netOutput) == 0)
                   converged = 1;
               else
                   iterations = iterations + 1;
                   netInput = netOutput;
               end
           end
           iterationVector(repetitionIdx) = iterations;
           overlapVector(repetitionIdx) = (netOutput*patternMatrix(testPatternIdx,:)')/networkSize;
       end
       
       if mean(overlapVector) > recallCriterion
           proportionRecalled(trainingPatternIdx) = proportionRecalled(trainingPatternIdx) + 1;
       end
       iterationMatrix(trainingPatternIdx,testPatternIdx) = mean(iterationVector);
       overlapMatrix(trainingPatternIdx,testPatternIdx) = mean(overlapVector);
   end
   
   proportionRecalled(trainingPatternIdx) = proportionRecalled(trainingPatternIdx)/trainingPatternIdx;
end

%% Plot the results of the simulation
clf,
subplot(2,2,1),imshow(iterationMatrix',[]),colorbar
title('#Iterations'),ylabel('Test pattern'),xlabel('#Trained patterns')
subplot(2,2,2),imshow(overlapMatrix', []),colorbar
title('Pattern overlap')
xlabel('#Trained patterns')
ylabel('Test pattern')

subplot(2,2,3),plot(1:nPatterns,weightValues(1,:),'r','LineWidth',2),hold on,title('Weight evolution')
plot(1:nPatterns,weightValues(2,:),'k','LineWidth',2),
plot(1:nPatterns,weightValues(3,:),'g','LineWidth',2)
xlabel('#Trained patterns')
ylabel('Weight value')
legend('Max','Mean','Min')
colormap jet
subplot(2,2,4),hold on
plot([0.14*networkSize 0.14*networkSize],[0 1],'r')
plot(1:nPatterns,proportionRecalled)
ylabel('Proportion recalled')
xlabel('# Trained patterns')
title('Proportion patterns recalled correctly')
legend('Theoretical capacity')
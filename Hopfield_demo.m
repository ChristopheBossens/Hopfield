%% Construct a default network and generate a pattern matrix
clc;clear
nPatterns = 32;
networkSize = 64;
hopfield = Hopfield(networkSize);

patternMatrix = hopfield.GeneratePatternMatrix(nPatterns);
%% Probe the capacity of the network.
% We add new patterns and after the addition of each new pattern we test if
% the network can still recall previous patterns. We keep track of how many
% steps it takes before the network converges and the overlap with the
% pattern that was used to probe the network

% Initialize data storage matrices
iterationMatrix = zeros(nPatterns);
overlapMatrix = zeros(nPatterns);
weightValues = zeros(3,nPatterns);
proportionRecalled = zeros(1,nPatterns);

% Set the simulation patterns
recallCriterion = 0.9;
nRepetitions = 5;
noiseLevel = .1;

hopfield.ResetWeightMatrix();
for trainingPatternIdx = 1:nPatterns
    % Add pattern and inspect statistics of weight matrix
   hopfield.StorePattern(patternMatrix(trainingPatternIdx,:));
   
   currentWeights = hopfield.GetWeightMatrix();
   weightValues(1,trainingPatternIdx) = max(currentWeights(:));
   weightValues(2,trainingPatternIdx) = mean(currentWeights(:));
   weightValues(3,trainingPatternIdx) = min(currentWeights(:));
    
   % Previous patterns are presented multiple times
   iterationVector = zeros(1,nRepetitions);
   overlapVector =  zeros(1,nRepetitions);
   for testPatternIdx = 1:trainingPatternIdx
       for repetitionIdx = 1:nRepetitions
           inputPattern = patternMatrix(testPatternIdx,:);
           initialState = hopfield.DistortPattern(inputPattern,noiseLevel);
           
           [finalState,it] = hopfield.Converge(initialState);
           
           iterationVector(repetitionIdx) = it;
           overlapVector(repetitionIdx) = (finalState*inputPattern')/networkSize;
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

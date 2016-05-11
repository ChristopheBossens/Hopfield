%% The following code corresponds to the paper:
% Gernuschi-Frias, B., & Segura, E.C.(1993). Concept learning in Hopfield
% associative memories trained with noisy examples using the Hebb rule.
% Proceedings of 1993 international Joint Conference on Neural Networks
clc;clear;
cd 'C:\Users\Christophe\Documents\GitHub\Hopfield'
hopfield = Hopfield;
networkSize = 64;
nExemplars = 64;
noiseLevel = 0.25;
nRepetitions = 10;
activityLevel = 0.5;

conceptPattern = hopfield.GeneratePattern(networkSize,activityLevel);
exemplarMatrix = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
   exemplarMatrix(exemplarIndex,:) = hopfield.DistortPattern(conceptPattern,noiseLevel); 
end

%% The network is trained on all individual exemplars. After adding each
% new pattern to the network, we test if the prototype is a stable pattern.
% We additionally probe the network for the existence of spurious states
iterationMatrix = zeros(nExemplars);
overlapMatrix = zeros(nExemplars);
weightValues = zeros(3,nExemplars);
conceptOverlapMatrix = zeros(nExemplars);

conceptIterationVector =zeros(1,nExemplars);
conceptOverlapVector = zeros(1,nExemplars);
spuriousStatesVector = zeros(1,nExemplars);

hopfield.ResetWeights();
for exemplarIndex = 1:nExemplars
    hopfield.AddPattern(exemplarMatrix(exemplarIndex,:),1);
    currentWeights = hopfield.GetWeightMatrix();
    weightValues(1,exemplarIndex) = max(currentWeights(:));
    weightValues(2,exemplarIndex) = mean(currentWeights(:));
    weightValues(3,exemplarIndex) = min(currentWeights(:));
    
    % Test recall for previously stored exemplars and how well the
    % converged state corresponds to the concept
    for testPatternIndex = 1:exemplarIndex
        iterationVector = zeros(1,nRepetitions);
        overlapVector = zeros(1,nRepetitions);
        
        for repetitionIndex = 1:nRepetitions
            [output,it] = hopfield.Converge(exemplarMatrix(testPatternIndex,:), 'async');
            iterationVector(repetitionIndex) = it;
            overlapVector(repetitionIndex) = (1/networkSize)*(exemplarMatrix(testPatternIndex,:)*output');
        end
        
        iterationMatrix(exemplarIndex,testPatternIndex) = mean(iterationVector);
        overlapMatrix(exemplarIndex, testPatternIndex) = mean(overlapVector);
        conceptOverlapMatrix(exemplarIndex, testPatternIndex) = (1/networkSize)*(conceptPattern*output');
    end
    
    % Test stability of concept prototype
    [output, it] = hopfield.Converge(conceptPattern, 'async');
    conceptIterationVector(exemplarIndex) = it;
    conceptOverlapVector(exemplarIndex) = (1/networkSize)*(conceptPattern*output');
    
    % Check existence of spurious states
    [stableStates, stateHist] = hopfield.GetSpuriousStates(100);
    spuriousStatesVector(exemplarIndex) = size(stableStates,1);
end
%% Plot the data
clf,
subplot(2,3,1),imshow(iterationMatrix',[]),colorbar,title('Iterations before converging')
subplot(2,3,2),imshow(overlapMatrix',[]),colorbar,title('Pattern overlap')
subplot(2,3,3),imshow(conceptOverlapMatrix',[]),colorbar,title('Overlap with concept')
subplot(2,3,4),hold on
plot(1:nExemplars,weightValues(1,:),'r','LineWidth',2)
plot(1:nExemplars,weightValues(2,:),'k','LineWidth',2)
plot(1:nExemplars,weightValues(3,:),'g','LineWidth',2)
subplot(2,3,5),plot(conceptOverlapVector),title('Concept pattern overlap')
subplot(2,3,6),plot(spuriousStatesVector),title('Spurious states detected')
colormap jet

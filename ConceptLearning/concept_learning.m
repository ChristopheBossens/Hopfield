%% The following code corresponds to the paper:
% Gernuschi-Frias, B., & Segura, E.C.(1993). Concept learning in Hopfield
% associative memories trained with noisy examples using the Hebb rule.
% Proceedings of 1993 international Joint Conference on Neural Networks
clc;clear;
cd 'C:\Users\Christophe\Documents\GitHub\Hopfield'
hopfield = Hopfield;
networkSize = 128;
nExemplars = 64;
noiseLevel = 0.3;
activityLevel = 0.5;

conceptPattern = hopfield.GeneratePattern(networkSize,activityLevel);
exemplarMatrix = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
   exemplarMatrix(exemplarIndex,:) = hopfield.DistortPattern(conceptPattern,noiseLevel); 
end

%% The network is trained on all individual exemplars. After adding each
% new pattern to the network, we test if the prototype is a stable pattern.
% We additionally probe the network for the existence of spurious states
overlapMatrix = zeros(nExemplars);
conceptOverlapMatrix = zeros(nExemplars);

conceptOverlapVector = zeros(1,nExemplars);
spuriousStatesVector = zeros(1,nExemplars);

hopfield.ResetWeights();
for exemplarIndex = 1:nExemplars
    hopfield.AddPattern(exemplarMatrix(exemplarIndex,:),1);
    
    % Test if the previously stored patterns are stable states, calculate
    % the overlap with the input pattern and with the prototype pattern
    for testPatternIndex = 1:exemplarIndex
        output = hopfield.Converge(exemplarMatrix(testPatternIndex,:), 'async');

        overlapMatrix(exemplarIndex, testPatternIndex) = (1/networkSize)*(exemplarMatrix(testPatternIndex,:)*output');
        conceptOverlapMatrix(exemplarIndex, testPatternIndex) = (1/networkSize)*(conceptPattern*output');
    end
    
    % Test stability of concept prototype
    [output, it] = hopfield.Converge(conceptPattern, 'async');
    conceptOverlapVector(exemplarIndex) = (1/networkSize)*(conceptPattern*output');
    
    % Check existence of spurious states
    [stableStates, stateHist] = hopfield.GetSpuriousStates(100);
    spuriousStatesVector(exemplarIndex) = size(stableStates,1);
end
%% Plot the data
clf,
subplot(2,2,1),imshow(overlapMatrix',[]),colorbar,title('Pattern overlap')
subplot(2,2,2),imshow(conceptOverlapMatrix',[]),colorbar,title('Overlap with concept')
subplot(2,2,3),plot(conceptOverlapVector),title('Concept pattern overlap')
subplot(2,2,4),plot(spuriousStatesVector),title('Spurious states detected')
set(gca,'YLim',[0 max(spuriousStatesVector)])
colormap jet

%% Same procedure, but with two different concept patterns
conceptPatternA = hopfield.GeneratePattern(networkSize,activityLevel);
conceptPatternB = hopfield.GeneratePattern(networkSize,activityLevel);
exemplarMatrixA = zeros(nExemplars,networkSize);
exemplarMatrixB = zeros(nExemplars,networkSize);

for exemplarIndex = 1:nExemplars
   exemplarMatrixA(exemplarIndex,:) = hopfield.DistortPattern(conceptPatternA,noiseLevel); 
   exemplarMatrixB(exemplarIndex,:) = hopfield.DistortPattern(conceptPatternB,noiseLevel);
end

% Seed network succesively with alternating A and B exemplars
overlapMatrix = zeros(nExemplars,nExemplars,2);
conceptOverlapMatrix = zeros(nExemplars,nExemplars,2);

conceptOverlapVector = zeros(2,nExemplars);
spuriousStatesVector = zeros(1,nExemplars);

hopfield.ResetWeights();
for exemplarIndex = 1:nExemplars
    hopfield.AddPattern(exemplarMatrixA(exemplarIndex,:),1);
    hopfield.AddPattern(exemplarMatrixB(exemplarIndex,:),1);
    
    for testPatternIndex = 1:exemplarIndex
        % A prototype
        output = hopfield.Converge(exemplarMatrixA(testPatternIndex,:), 'async');
        overlapMatrix(exemplarIndex, testPatternIndex,1) = (1/networkSize)*(exemplarMatrixA(testPatternIndex,:)*output');
        conceptOverlapMatrix(exemplarIndex, testPatternIndex,1) = (1/networkSize)*(conceptPatternA*output');
        
        % B prototype
        output = hopfield.Converge(exemplarMatrixB(testPatternIndex,:), 'async');
        overlapMatrix(exemplarIndex, testPatternIndex,2) = (1/networkSize)*(exemplarMatrixB(testPatternIndex,:)*output');
        conceptOverlapMatrix(exemplarIndex, testPatternIndex,2) = (1/networkSize)*(conceptPatternB*output');
    end
    
    % Test stability of concept prototype
    output = hopfield.Converge(conceptPatternA, 'async');
    conceptOverlapVector(exemplarIndex,1) = (1/networkSize)*(conceptPatternA*output');
    
    output = hopfield.Converge(conceptPatternB, 'async');
    conceptOverlapVector(exemplarIndex,2) = (1/networkSize)*(conceptPatternB*output');
    
    
    % Check existence of spurious states
    [stableStates, stateHist] = hopfield.GetSpuriousStates(100);
    spuriousStatesVector(exemplarIndex) = size(stableStates,1);
end
%% Plot the results
clf,
subplot(2,3,1),imshow(overlapMatrix(:,:,1)',[]),colorbar,title('Pattern A overlap')
subplot(2,3,2),imshow(conceptOverlapMatrix(:,:,1)',[]),colorbar,title('Overlap with concept A')
subplot(2,3,4),imshow(overlapMatrix(:,:,2)',[]),colorbar,title('Pattern B overlap')
subplot(2,3,5),imshow(conceptOverlapMatrix(:,:,2)',[]),colorbar,title('Overlap with concept B')

subplot(1,3,3),plot(spuriousStatesVector),title('Spurious states detected')
set(gca,'YLim',[0 max(spuriousStatesVector)])
colormap jet
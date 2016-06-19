%% The following code corresponds to the paper:
% Builds on the concept learning paper, and compares network statistics for
% random patterns vs a distorted concept pattern
clc;clear;
cd 'C:\Users\Christophe\Documents\GitHub\Hopfield'
hopfieldConcept = Hopfield;
hopfieldRandom  = Hopfield;
networkSize = 128;
nExemplars = 64;
noiseLevel = 0.3;
activityLevel = 0.5;
thresholdActivity = .9;

%% The network is trained on all individual exemplars. After adding each
% new pattern to the network, we test if the prototype is a stable pattern.
% We additionally probe the network for the existence of spurious states
nRepetitions = 10;
capacityVector = zeros(2,nExemplars);
spuriousStatesVector = zeros(2,nExemplars);

weightBins = -40:2:40;
weightsConcept = zeros(length(weightBins),nExemplars);
weightsRandom = zeros(length(weightBins),nExemplars);

for repetitionIndex = 1:nRepetitions  
    clc;display(['Progress: ' num2str(repetitionIndex) '/' num2str(nRepetitions)]);
    hopfieldConcept.ResetWeights();
    hopfieldRandom.ResetWeights();
    
    
    conceptPattern = hopfieldConcept.GeneratePattern(networkSize,activityLevel);
    exemplarMatrixConcept = zeros(nExemplars,networkSize);
    exemplarMatrixRandom = zeros(nExemplars,networkSize);
    for exemplarIndex = 1:nExemplars
       exemplarMatrixConcept(exemplarIndex,:) = hopfieldConcept.DistortPattern(conceptPattern,noiseLevel); 
       exemplarMatrixRandom(exemplarIndex,:)  = hopfieldRandom.GeneratePattern(networkSize, activityLevel);
    end

    for exemplarIndex = 1:nExemplars
        % Add next pattern and record the weights
        hopfieldConcept.AddPattern(exemplarMatrixConcept(exemplarIndex,:),1);
        hopfieldRandom.AddPattern(exemplarMatrixRandom(exemplarIndex,:),1);
        
        A = hopfieldConcept.GetWeightMatrix();
        weightsConcept(:,exemplarIndex) = weightsConcept(:,exemplarIndex) + histc(A(:),weightBins);
        B = hopfieldRandom.GetWeightMatrix();
        weightsRandom(:,exemplarIndex) = weightsRandom(:,exemplarIndex) + histc(B(:),weightBins);
        
        % Test if the previously stored patterns are stable states, calculate
        % the overlap with the input pattern and with the prototype pattern
        capacity = zeros(1,2);
        for testPatternIndex = 1:exemplarIndex
            outputConcept = hopfieldConcept.Converge(exemplarMatrixConcept(testPatternIndex,:), 'async');
            outputRandom = hopfieldRandom.Converge(exemplarMatrixRandom(testPatternIndex,:), 'async');
            
            conceptOverlap = (1/networkSize)*(exemplarMatrixConcept(testPatternIndex,:)*outputConcept');
            randomOverlap = (1/networkSize)*(exemplarMatrixRandom(testPatternIndex,:)*outputRandom');
            if conceptOverlap > thresholdActivity
                capacity(1) = capacity(1) + 1;
            end
            if randomOverlap > thresholdActivity
                capacity(2) = capacity(2) + 1;
            end
        end
        
        capacityVector(1,exemplarIndex) = capacityVector(1,exemplarIndex) + capacity(1)/exemplarIndex;
        capacityVector(2,exemplarIndex) = capacityVector(2,exemplarIndex) + capacity(2)/exemplarIndex;
        
        % Check existence of spurious states
        [stableStates] = hopfieldConcept.GetSpuriousStates(100);
        spuriousStatesVector(1, exemplarIndex) = spuriousStatesVector(1, exemplarIndex) + size(stableStates,1);
        
        [stableStates] = hopfieldRandom.GetSpuriousStates(100);
        spuriousStatesVector(2, exemplarIndex) = spuriousStatesVector(2, exemplarIndex) + size(stableStates,1);
    end
end

capacityVector = capacityVector./nRepetitions;
spuriousStatesVector = spuriousStatesVector./nRepetitions;
weightsConcept = weightsConcept./nRepetitions;
weightsRandom = weightsRandom./nRepetitions;
%% Plot the data
weightsMin = min([weightsConcept(:); weightsRandom(:)]);
weightsMax = max([weightsConcept(:); weightsRandom(:)]);

clf,
subplot(2,2,1),hold on
plot(capacityVector(1,:),'g','LineWidth',2)
plot(capacityVector(2,:),'r','LineWidth',2)
legend('Concept','Random')
title('Network capacity')
subplot(2,2,2),hold on
plot(spuriousStatesVector(1,:),'g','LineWidth',2)
plot(spuriousStatesVector(2,:),'r','LineWidth',2)
legend('Concept','Random')
title('Spurious states')
subplot(2,2,3),
imshow(weightsConcept,[weightsMin weightsMax]),title('Concept')
subplot(2,2,4),
imshow(weightsRandom,[weightsMin weightsMax]),title('Random')
colormap jet
%% The network is trained on individual patterns. During different parts of
% training we investigate how sensitive the network is to pruning vs random
% weight removal
%% The following code corresponds to the paper:
% Gernuschi-Frias, B., & Segura, E.C.(1993). Concept learning in Hopfield
% associative memories trained with noisy examples using the Hebb rule.
% Proceedings of 1993 international Joint Conference on Neural Networks
clc;clear;
networkSize = 128;
nExemplars = 64;

hopfield = Hopfield(networkSize);
noiseLevel = 0.3;
activityLevel = 0.5;

conceptPattern = hopfield.GeneratePattern(activityLevel);
exemplarMatrix = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
   exemplarMatrix(exemplarIndex,:) = hopfield.DistortPattern(conceptPattern,noiseLevel); 
end

%% The network is trained on all individual exemplars. After adding each
% new pattern to the network, we test if the prototype is a stable pattern.
% We additionally probe the network for the existence of spurious states
conceptOverlapVector = zeros(1,nExemplars);
exemplarOverlapVector = zeros(1,nExemplars);

hopfield.ResetWeightMatrix();
for exemplarIndex = 1:nExemplars
    hopfield.StorePattern(exemplarMatrix(exemplarIndex,:),1/networkSize);
    
    % Test if the previously stored patterns are stable states, calculate
    % the overlap with the input pattern and with the prototype pattern
    for testPatternIndex = 1:exemplarIndex
        input = exemplarMatrix(testPatternIndex,:);
        inputDistorted = hopfield.DistortPattern(input, 0.2);
        output = hopfield.Converge(inputDistorted);
        exemplarOverlapVector(exemplarIndex) = exemplarOverlapVector(exemplarIndex) + (input*output')/networkSize;
    end
    exemplarOverlapVector(exemplarIndex) = exemplarOverlapVector(exemplarIndex)./exemplarIndex;
    
    % Test stability of concept prototype
    [output, it] = hopfield.Converge(conceptPattern);
    conceptOverlapVector(exemplarIndex) = (conceptPattern*output')/networkSize;
end
%% Plot the data
clf,hold on
plot(1:nExemplars,conceptOverlapVector,'--g','LineWidth',2),
plot(1:nExemplars,exemplarOverlapVector,'--r','LineWidth',2)
title('Concept vs exemplar overlap'),xlabel('# Exemplars added'),ylabel('Prototype pattern overlap')
legend('Concept overlap','Exemplar overlap')
colormap jet

%% Same procedure, but with two different concept patterns
conceptPatternA = hopfield.GeneratePattern(activityLevel);
conceptPatternB = hopfield.GeneratePattern(activityLevel);

exemplarMatrixA = zeros(nExemplars,networkSize);
exemplarMatrixB = zeros(nExemplars,networkSize);
for exemplarIndex = 1:nExemplars
   exemplarMatrixA(exemplarIndex,:) = hopfield.DistortPattern(conceptPatternA,noiseLevel); 
   exemplarMatrixB(exemplarIndex,:) = hopfield.DistortPattern(conceptPatternB,noiseLevel);
end


conceptOverlapVector = zeros(2,nExemplars);
exemplarOverlapVector = zeros(2,nExemplars);
hopfield.ResetWeightMatrix();
for exemplarIndex = 1:nExemplars
    hopfield.StorePattern(exemplarMatrixA(exemplarIndex,:),1);
    hopfield.StorePattern(exemplarMatrixB(exemplarIndex,:),1);
    
    for testPatternIndex = 1:exemplarIndex
        % A prototype
        input = exemplarMatrixA(testPatternIndex,:);
        output = hopfield.Converge(hopfield.DistortPattern(input,0.2));
        exemplarOverlapVector(1,exemplarIndex) = exemplarOverlapVector(1,exemplarIndex) + (input*output')/networkSize;
        
        % B prototype
        input = exemplarMatrixB(testPatternIndex,:);
        output = hopfield.Converge(hopfield.DistortPattern(input,0.2));
        exemplarOverlapVector(2,exemplarIndex) = exemplarOverlapVector(2,exemplarIndex) + (input*output')/networkSize;
    end
    exemplarOverlapVector(:,exemplarIndex) = exemplarOverlapVector(:,exemplarIndex)/exemplarIndex;
    
    % Test stability of concept prototype
    output = hopfield.Converge(hopfield.DistortPattern(conceptPatternA,0.1));
    conceptOverlapVector(1,exemplarIndex) = (conceptPatternA*output')/networkSize;
    
    output = hopfield.Converge(hopfield.DistortPattern(conceptPatternB,0.1));
    conceptOverlapVector(2,exemplarIndex) = (conceptPatternB*output')/networkSize;
end
%% Plot the results
clf,
subplot(1,2,1),hold on
plot(1:nExemplars,conceptOverlapVector(1,:),'--g','LineWidth',2),
plot(1:nExemplars,exemplarOverlapVector(1,:),'--r','LineWidth',2),
title('A'),xlabel('# Exemplars added'),ylabel('Prototype pattern overlap')

subplot(1,2,2),hold on
plot(1:nExemplars,conceptOverlapVector(2,:),'--g','LineWidth',2),
plot(1:nExemplars,exemplarOverlapVector(2,:),'--r','LineWidth',2),
title('B'),xlabel('# Exemplars added'),ylabel('Prototype pattern overlap')
colormap jet
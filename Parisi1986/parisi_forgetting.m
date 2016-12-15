%% The following code implements the method discussed in:
% Parisi, G. (1986). A memory which forgets. Journal of Physics A:
% Mathematical and General
clc;clear;
nPatterns = 40;

%% Run the simulation
nRepetitions = 10;
recallCriterion = 0.98;
noiseLevel = .1;
networkSizes = [100 200 300];
nNetworkSizes = length(networkSizes);
clipValues = 0.05:0.05:1.25;
nClipValues = length(clipValues);

recallPerformance = zeros(nNetworkSizes,nClipValues);
for repetitionIndex = 1:nRepetitions
    for networkSizeIndex = 1:nNetworkSizes
        hopfield = Hopfield(networkSizes(networkSizeIndex));
        hopfield.SetUpdateMode('sync');
        patternMatrix = hopfield.GeneratePatternMatrix(nPatterns);
        for clipIndex = 1:nClipValues
            display(['Testing repetition ' num2str(repetitionIndex) '/' num2str(nRepetitions)])
            display(['networkSize: ' num2str(networkSizes(networkSizeIndex))])
            display(['Clip value: ' num2str(clipValues(clipIndex))])
            hopfield.ResetWeightMatrix();
            hopfield.EnableWeightClipping(clipValues(clipIndex));
            hopfield.StorePatternMatrix(patternMatrix,sqrt(1/networkSizes(networkSizeIndex)));

           % Test recollection on all previously learned patterns
           for testPatternIdx = 1:nPatterns
               input = patternMatrix(testPatternIdx,:);
               distortedInput = hopfield.DistortPattern(input, noiseLevel);
               output = hopfield.Converge(distortedInput);

               if (sum(input==output)==networkSizes(networkSizeIndex)) > recallCriterion
                    recallPerformance(networkSizeIndex,clipIndex) = recallPerformance(networkSizeIndex,clipIndex) + 1;
               end
           end
        end
    end
end

for i = 1:nNetworkSizes
    recallPerformance(i,:) = recallPerformance(i,:)./(networkSizes(i)*nRepetitions);
end
%% Plot the results of the simulation
symbols = {'.-','*-','o-'};
clf, hold on
for i = 1:nNetworkSizes
    plot(clipValues,recallPerformance(i,:),symbols{i})
end
xlabel('Clip value')
ylabel('Proportion recalled')
legend('100','200','300')
title('Optimal clip value')
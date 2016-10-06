%% Set the simulation parameters
clear
networkSize = 32;
nPatterns = 4;
nStateSamples = 1000;
patternActivity = 0.5;
deltaTrainingEpochs = 100;

% Select simulation type
% 1: Standard Hebbian learning
% 2: Delta learning
% 3: Delta learning with noise
simulationType = 3;

hopfield = Hopfield(networkSize);
patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);

switch simulationType
    case 1
        hopfield.AddPatternMatrix(patternMatrix,1);
    case 2
        hopfield.LearnDeltaPatterns(patternMatrix,0.5,deltaTrainingEpochs);
    case 3
        hopfield.SetInputNoise(0,networkSize);
        hopfield.LearnDeltaPatterns(patternMatrix,0.5,deltaTrainingEpochs);
end
hopfield.SetInputNoise(0,0);

[stablePatternMatrix, stablePatternCount, meanPatternIterations] = hopfield.SampleWithRandomStates(nStateSamples);
[isSpuriousState, oldPatternIndex] = hopfield.AnalyseStableStates(stablePatternMatrix,patternMatrix);
nStableStates = length(stablePatternCount);
%% Construct figure 2 of the paper
% Get the sorted unit energy
unitEnergy = zeros(nStableStates,networkSize);

for i = 1:nStableStates
   unitEnergy(i,:) = sort(hopfield.GetUnitEnergy(stablePatternMatrix(i,:))); 
end

mOld = mean(unitEnergy(isSpuriousState==0,:));
mSpurious = mean(unitEnergy(isSpuriousState==1,:));
eOld = std(unitEnergy(isSpuriousState==0,:));
eSpurious = std(unitEnergy(isSpuriousState == 1,:));

% Produce the picture
clf,
hold on
plot(1:networkSize,mOld,'g','LineWidth',2)
plot(1:networkSize,mSpurious,'r', 'LineWidth',2)
errorbar(1:networkSize,mOld,eOld,'.g')
errorbar(1:networkSize,mSpurious,eSpurious,'.r')
xlabel('Unit')
ylabel('Unit energy')
title('Sorted unit energy')
legend('Memory','Spurious','Location','SouthEast')

%% Construct figure 3 of the paper
% Calculate the energy ratio (needs the unit energy from figure 2)
energyRatio = zeros(nStableStates,1);
for i = 1:nStableStates
   energyRatio(i) = sum(unitEnergy(i,1:3))/sum(unitEnergy(i,(end-2):end));
end
[temp,sortedIndex] = sort(isSpuriousState);
patternLabel = cell(1,nStableStates);
lIndex = 1;
sIndex = 1;
for i = 1:nStableStates
    if temp(i) == 0
        patternLabel{i} = ['L' num2str(lIndex)];
        lIndex = lIndex + 1;
    else
        patternLabel{i} = ['S' num2str(sIndex)];
        sIndex = sIndex + 1;
    end
end

% Produce the picture
clf,
bar(1:nStableStates,energyRatio(sortedIndex))
xlabel('Pattern')
ylabel('Energy ratio')
title('Stable state energy ratio')
set(gca,'XTick',[1:nStableStates],'XTickLabel',patternLabel)
%% Assess the performance of energy ratio based discrimination
% We perform multiple simulations in which we incrementally add new
% patterns to the network and then test the performance of the
% classification based on energy ratios
clc;clear;
nPatterns = 10;
networkSize = 32;
nProbes = 500;
deltaTrainingEpochs = 40;
nSimulations = 3;

classificationData = zeros(nSimulations,3,nPatterns);
for simulationIndex = 1:nSimulations
    hopfield = Hopfield(networkSize);
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);

    for patternIndex = 1:nPatterns
        for networkType = 1:3
            clc;
            display(['Simulation ' num2str(simulationIndex) '/' num2str(nSimulations)])
            display(['Network load : ' num2str(patternIndex/networkSize)]);
            display(['Network type: ' num2str(networkType)]);
            % Learn the given number of patterns
            hopfield.ResetWeightMatrix();
            switch networkType
                case 1
                    hopfield.AddPatternMatrix(patternMatrix(1:patternIndex,:),1);
                case 2
                    hopfield.LearnDeltaPatterns(patternMatrix(1:patternIndex,:),0.5,deltaTrainingEpochs);
                case 3
                    hopfield.SetInputNoise(0,networkSize/2);
                    hopfield.LearnDeltaPatterns(patternMatrix(1:patternIndex,:),0.5,deltaTrainingEpochs);
            end            
            hopfield.SetInputNoise(0,0);
            
            % Get the lowest energy ratio
            energyRatio = zeros(1,patternIndex);
            for i = 1:patternIndex
               unitEnergy = sort(hopfield.GetUnitEnergy(patternMatrix(i,:))); 
               energyRatio(i) = sum(unitEnergy(1:3))/sum(unitEnergy((end-2):end));
            end
            energyCriterion = min(energyRatio);

            % Probe the network with random probes and check which stable
            % states correspond to spurious states or stored patterns
            [stablePatternMatrix, stablePatternCount, meanPatternIterations] = hopfield.SampleWithRandomStates(nProbes);
            [isSpuriousState, oldPatternIndex] = hopfield.AnalyseStableStates(stablePatternMatrix,patternMatrix(1:patternIndex,:));

            % Test which of the stable states can be classified correctly
            % according to the energy ratio criterion
            correctCount = 0;
            for j = 1:size(stablePatternMatrix,1)
                unitEnergy = sort(hopfield.GetUnitEnergy(stablePatternMatrix(j,:)));
                energyRatio = sum(unitEnergy(1:3))/sum(unitEnergy((end-2):end));

                if ((energyRatio < energyCriterion) && (isSpuriousState(j) == 1)) || ...
                        ((energyRatio >= energyCriterion) && (isSpuriousState(j) == 0))
                    correctCount = correctCount + 1;
                end
            end
            
            classificationData(simulationIndex,networkType,patternIndex) = correctCount/size(stablePatternMatrix,1);
        end
    end
end

%%
a = squeeze(mean(classificationData));
e = squeeze(std(classificationData));
x = (1:nPatterns)./networkSize;
clf,hold on
plot( x,a(1,:),'k','LineWidth',2)
plot( x,a(2,:),'b','LineWidth',2)
plot( x,a(3,:),'g','LineWidth',2)
errorbar(x,a(1,:),e(1,:),'.k','LineWidth',2)
errorbar(x,a(2,:),e(2,:),'.b','LineWidth',2)
errorbar(x,a(3,:),e(3,:),'.g','LineWidth',2)
legend('Hebbian','Delta','Delta + noise')
title('Spurious state classification performance')
ylabel('Proportion correct')
xlabel('Network load')
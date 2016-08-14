clear
% Define simulation parameters and configure hopfield network
networkSize = 100;
nPatterns = 40;
gamma = 1.05;
epsilon = 0.1;
noiseLevel = 0.2;
nSimulations = 10;

hopfield = Hopfield(networkSize);
hopfield.SetGamma(gamma);
hopfield.EnableWeightClipping(1);
%% Run simulations for obtaining weight distributions for figure 3
bins = -1:0.2:1;
weightDistribution = zeros(nPatterns,length(bins));
for simulationIndex = 1:nSimulations
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    hopfield.ResetWeights();
    for patternIndex = 1:nPatterns  
       hopfield.AddPattern(patternMatrix(patternIndex,:),epsilon);
       
       weightMatrix = hopfield.GetWeightMatrix();
       dist = hist(weightMatrix(:),bins);
       weightDistribution(patternIndex,:) = weightDistribution(patternIndex,:) + dist./sum(dist);
    end
end
weightDistribution = weightDistribution./nSimulations;
%% Code to reproduce the weight distributions from figure 3
clf,hold on
plot(bins,weightDistribution(10,:),'-k','LineWidth',2)
plot(bins,weightDistribution(20,:),'--k','LineWidth',2)
plot(bins,weightDistribution(30,:),':k','LineWidth',2)
title('Weight distribution'),xlabel('Weight value'),ylabel('Proportion')
legend('10 patterns','20 patterns','30 patterns')

%% Simulations for producing the serial position curves in figure 4/5/6
selectedFigure = 6;
switch selectedFigure
    case 4
        gamma = 1.25;
        epsilon = 0.2;
    case 5
        gamma = 1.05;
        epsilon= 0.45;
    case 6
        gamma = 1.14;
        epsilon = 0.3;
end

overlapMatrix = zeros(nPatterns,nPatterns);
for simulationIndex = 1:nSimulations
    hopfield.ResetWeights();
    hopfield.SetGamma(gamma);
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    for patternIndex = 1:nPatterns
        hopfield.AddPattern(patternMatrix(patternIndex,:),epsilon);
        for testIndex = 1:patternIndex
            originalPattern = patternMatrix(testIndex,:);
            testPattern = hopfield.DistortPattern(originalPattern,noiseLevel);
            finalState = hopfield.Converge(testPattern);
            overlapMatrix(testIndex,patternIndex) = overlapMatrix(testIndex,patternIndex) + (originalPattern*finalState')/networkSize;
        end
    end
end
overlapMatrix = overlapMatrix./nSimulations;
%% Code for generating figure 4/5/6
clf,hold on
plot(1:10,overlapMatrix(1:10,10),'-k','LineWidth',2)
plot(1:20,overlapMatrix(1:20,20),'--k','LineWidth',2)
plot(1:30,overlapMatrix(1:30,30),':k','LineWidth',2)
legend('10 patterns','20 patterns','30 patterns')
xlabel('N° trained patterns')
ylabel('Pattern overlap')
title(['Serial position curve (\gamma = ' num2str(gamma) ', \epsilon = ' num2str(epsilon)])

clear
% Define simulation parameters and configure hopfield network
networkSize = 100;
nPatterns = 40;
noiseLevel = 0.15;
nSimulations = 10;

hopfield = Hopfield(networkSize);
hopfield.EnableWeightClipping(1);
%% Run simulations for obtaining weight distributions for figure 3
% parameters used in the paper:
gamma = 1.05;
epsilon = 0.1;
hopfield.SetGamma(gamma);

bins = -1:0.2:1;
weightDistribution = zeros(nPatterns,length(bins));
for simulationIndex = 1:nSimulations
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    hopfield.ResetWeightMatrix();
    for patternIndex = 1:nPatterns  
       hopfield.StorePattern(patternMatrix(patternIndex,:),epsilon);
       
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
hopfield.SetUpdateMode('sync');
for simulationIndex = 1:nSimulations
    hopfield.ResetWeightMatrix();
    hopfield.SetGamma(gamma);
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    for patternIndex = 1:nPatterns
        hopfield.StorePattern(patternMatrix(patternIndex,:),epsilon);
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

%% Simulation code for figure 7
networkSize = 700;
epsilon = 0.2;
noiseLevel = 0.15;
nSimulations = 10;

rpiHopfield = Hopfield(networkSize);
rpiHopfield.EnableWeightClipping(1);
controlHopfield = Hopfield(networkSize);
controlHopfield.EnableWeightClipping(1);

nCategories = 4;
nTrials = 24;
nMembersPerTrial = 3;

rpiOverlapData = zeros(nSimulations,nTrials);
controlOverlapData = zeros(nSimulations,nTrials);

for simulationIndex = 1:nSimulations
    % Generate the category patterns for the rpi experiment
    prototypePatterns = zeros(nCategories,networkSize);
    for i = 1:nCategories
        prototypePatterns(i,:) = rpiHopfield.GeneratePattern();
    end
    
    % Generate prototype patterns for the control experiment
    controlPrototypePatterns = zeros(nTrials,networkSize);
    for i = 1:nTrials
        controlPrototypePatterns(i,:) = controlHopfield.GeneratePattern();
    end
    
    % Generate the exemplar patterns from the prototype patterns
    rpiPrototypeIndex = 1;
    controlPrototypeIndex = 1;
    nPatterns = nTrials*nMembersPerTrial;
    rpiPatternMatrix = zeros(nPatterns,networkSize);
    controlPatternMatrix = zeros(nPatterns,networkSize);
    for i = 1:nPatterns
        rpiPatternMatrix(i,:) = rpiHopfield.DistortPattern(prototypePatterns(rpiPrototypeIndex,:),194/networkSize);
        controlPatternMatrix(i,:) = controlHopfield.DistortPattern(controlPrototypePatterns(controlPrototypeIndex,:),194/networkSize);
        
        if mod(i,(nTrials/nCategories)*nMembersPerTrial)==0
            rpiPrototypeIndex = rpiPrototypeIndex + 1;
        end
        
        if mod(i,nMembersPerTrial) == 0
            controlPrototypeIndex = controlPrototypeIndex + 1;
        end
    end

    rpiHopfield.ResetWeightMatrix();
    controlHopfield.ResetWeightMatrix();
    for i = 1:nPatterns
        % Add patterns
        rpiHopfield.StorePattern(rpiPatternMatrix(i,:),epsilon);
        controlHopfield.StorePattern(controlPatternMatrix(i,:),epsilon);
        
        % Test the network every 3 patterns (i.e. a single trial)
        if mod(i,nMembersPerTrial) == 0
            for j = (i-(nMembersPerTrial-1)):i
                originalPattern = rpiPatternMatrix(j,:);
                distortedPattern = rpiHopfield.DistortPattern(originalPattern,noiseLevel);
                finalState = rpiHopfield.Converge(distortedPattern);
                rpiOverlapData(simulationIndex,ceil(i/nMembersPerTrial)) = rpiOverlapData(simulationIndex,ceil(i/nMembersPerTrial)) + (originalPattern*finalState')/networkSize;
                
                originalPattern = controlPatternMatrix(j,:);
                distortedPattern = controlHopfield.DistortPattern(originalPattern, noiseLevel);
                finalState = controlHopfield.Converge(distortedPattern);
                controlOverlapData(simulationIndex,ceil(i/nMembersPerTrial)) = controlOverlapData(simulationIndex,ceil(i/nMembersPerTrial)) + (originalPattern*finalState')/networkSize;
            end
        end
    end
    rpiOverlapData(simulationIndex,:) = rpiOverlapData(simulationIndex,:)./nMembersPerTrial;
    controlOverlapData(simulationIndex,:) = controlOverlapData(simulationIndex,:)./nMembersPerTrial;
end
%%
clf,hold on
plot(mean(rpiOverlapData),'-k','LineWidth',2)
plot(mean(controlOverlapData),'-.k','LineWidth',2)
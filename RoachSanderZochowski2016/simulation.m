N = 100;
P = 15;

h = Hopfield(N);
p = h.GeneratePatternMatrix(P,0.5);
h.AddPatternMatrix(p,1/N);

% Get some network statistics
[counts, bins, potentialValues, activityValues] = h.GetPotentialDistribution(100);
w = h.GetWeightMatrix();
clf,
subplot(1,2,1)
hold on
bar(bins,counts(1,:))
bar(bins,counts(2,:))
subplot(1,2,2),
hist(w(:))
%% Different temperature functions
x = -1.5:0.01:1.5;
T1 = 0.1;
T2 = 1/T1;
Pt = 1./(1 + exp(-2.*x./T1));
Pd = 0.5.*(1+tanh(T2.*x));
clf,hold on
plot(x,Pt)
plot(x,Pd)
legend('Exponential','TanH')
%% Code for reproducing figure 1b
N = 1000;
h = Hopfield(N);
nSteps = 50;

alphaValues = 0.01:0.01:0.25;
tValues = [0.001:0.002:0.17];
nAlpha = length(alphaValues);
nT = length(tValues);

results = zeros(nT,nAlpha);
for alphaIndex = 1:nAlpha
    alpha = alphaValues(alphaIndex);
    P = round(alpha*N);
    
    % Generate pattern matrix with strong and weak patterns
    p = h.GeneratePatternMatrix(P,0.5);
    w = ones(1,P);
    w(1) = 3;
    h.ResetWeightMatrix()
    for i = 1:P
        h.AddPattern(p(i,:),w(i)/(N*sum(w)));
    end
    
    for tIndex = 1:nT
        % Test different patterns for a fixed amount of steps
        h.UseStochasticDynamics(1/tValues(tIndex));
        patternId = 2;
        initialState = p(patternId,:);
        for i = 1:nSteps
            finalState = h.Iterate(initialState);
            initialState = finalState;
        end
        strongOverlap = (finalState*p(1,:)')/N;
        weakOverlap   = (finalState*p(patternId,:)')/N;
        results(tIndex,alphaIndex) = abs(strongOverlap)-abs(weakOverlap);
    end  
end
%% Create the figure for experiment 1b
clf,
imshow(results,[-1 1],'InitialMagnification','fit'),colormap hot, colorbar
axis on, axis square
set(gca,'XTick',[1:7:nAlpha],'XTickLabel',alphaValues(1:7:nAlpha))
set(gca,'YTick',[1:10:nT],'YTickLabel',tValues(1:10:nT),'YDir','normal')
xlabel('Saturation'),ylabel('Temperature'),
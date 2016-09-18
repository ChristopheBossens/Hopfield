# Low activity patterns
Hopfield networks are typically introduced using patterns in which approximately half of the units are active. If the Hopfield network is taken as a model for an actual brain, this property typically does not respond to biological reality. Here, it is often found that different cortical areas can exhibit sparse activation patterns. The question then becomes if we can construct Hopfield networks that can also store sparse activity patterns. This is possible, but in order to do this the pattern values need to be adjusted by subtracting the mean activity level:

```matlab
networkSize = 300;
p = 0.1;	% Probability of having an active unit

exemplar = hopfield.GeneratePattern(networkSize,p);
hopfield.AddPattern(exemplar - mean(exemplar), 1/networkSize);
```

The script in this folder performs a simulation that tests the capacity of a Hopfield network when patterns with different activity levels are stored in the network. We use the definition that the network has reached capacity as soon as one of the previously stored patterns is no longer a stable state of the network. At the same time, we use different threshold values for each neuron. The results show that not only the patterns need to be adjusted, but it is also important to use a proper threshold value:

![Concept learning](/LowActivity/adjustedPatterns.jpg?raw=true)

The reason for choosing an adjusted threshold value is illustrated in the following plot. When active and inactive units are balanced, no threshold is needed because the input potential distribution of active and inactive units can be separated using a threshold centered at zero:

![Concept learning](/LowActivity/thresholdp5.jpg?raw=true)

However, if we adjust the patterns we can observe that the location of these distributions shift. For example, in the case of low activity patterns the distribution of input potentials of units that should be inactive shifts to the right. The implication is that if we now still use zero as a threshold value, some of these inactive units will become activated. In the case of low activity patterns, adding a positive threshold solves the problem

![Concept learning](/LowActivity/thresholdp1.jpg?raw=true)
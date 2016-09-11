# Neural network models of list learning
Burgess, N., Shapiro, J., & Moore, M. A. (1991). Network, 2, 399-422

## Paper summary
A classical result in human memory research is the serial position effect that is obtained when subjects are first presented a list of items and are subsequently askes to recall these items. More specifically, the serial position effect is a u shaped curve for recall accuracy as a function of the position of an item in the list. Items that are presented first and last are recalled better than items in the middle. Better recall of items presented first is also refered to as the primacy effect, while recall of items presented last is also known as the recency effect.

In its original form, the Hopfield model is unable to account for these effects. When new patterns are learned, the model initially recalls all pattterns equally well up to a specific number. After this, performance decreases dramatically and none of the stored patterns can be recalled any more. Parisi (1986) used a modified learning rule which clips the weights at a specific value. Using these learning rule, new patterns can be added to the network. For a small number of patterns, all patterns can be recalled. When a capacity limit is reached, old patterns are forgotten and only the most recent patterns are correctly recalled. This behavior accounts for the recency effect.

The current paper expands this work by adding another term to the learning rule that has the effect of reinforcing the existing weight matrix. Together with a clipping rule for the weights, the model is able to show both primacy and recency effects.

## Simulation results
This figure shows how the weights evolve as a function of patterns that are learned

![Weight distribution](/BurgessShapiroMoore1991/figure3_a.jpg?raw=true)

The next figure show the primacy effect only. After a certain number of patterns is stored, no additional patterns can be recalled by the network. Initially learned patterns however can be recalled.

![Spike frequency adaptation](/BurgessShapiroMoore1991/figure4.jpg?raw=true)

Removing the reinforcement of the weight matrix produces a model which only shows a recency effect

![Spike frequency adaptation](/BurgessShapiroMoore1991/figure5.jpg?raw=true)

Enabling both learning adjustments at the same time produces a model which show both primacy and recency effects.
![Spike frequency adaptation](/BurgessShapiroMoore1991/figure6.jpg?raw=true)
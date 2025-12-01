## Summary

Across the three splits there is a mix of stable clips with no label changes and clips with multiple
ready ↔ not_ready transitions. There are no single transition clips, so the dataset mostly covers either
static behavior or more complex mealtime patterns. Validation clips all have multiple transitions, while
train and test each include a few no_transition clips. At the window level the train split is skewed
toward not_ready (about 64 percent) and the test split is skewed toward ready (about 74 percent). This
imbalance is moderate, so the Random Forest will be trained without class weights, but the neural
network will later use class weighting that upweights READY on the train split.



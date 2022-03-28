## TODOs
- [ ] Implement a tree policy function in mcts node_to_leaf
- [x] Reset board after game is over
- [x] Fortsette refactoring (reformat)
- [x] Not allowed to overwrite another players piece on the board
- [x] Add player and position as parameters to find_neighbours function

## Questions to studass
- [ ] How to represent a state in MCT
- [ ] Does reformat board work with np.where(Piece.player == 1) ?
- [ ] Should we rollout from all children, or follow tree-policy in choosing this as well?
- [ ] How should we represent board for input to anet?

- [x] SparseCategorialCrossentropy or CategoricalCrossentropy? Should we use one-hot encoding?
- [x] How to know when game is over? A*? Keep track of neighbours?

- [x] Can we use same initializer for weights and biases in neural net
- [x] What values should we use in the initializer
- [x] Should we initialize weights and biases in input/output layer as well?
- [x] Should we use MeanSquaredError as loss function? -> cross-entropy

## Tips from studass
- State: add one for which player makes the move
- Loss: use cross-entropy - this is also stated in the project paper
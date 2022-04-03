## TODOs
- [ ] argmax og argmin??? trene best and bad?
- [ ] debug visit = true/false in mcts - reset after each iteration?
- [x] M = num_cashed instead of save_interval
- [ ] Implement Tournament
- [ ] Debugging -> do we get improvement?
- [ ] Train models
- [ ] Tweak parameters (opt, act, hidden_layers)
- [x] Create a rollout func
- [x] Get node from state to use for visit_dist (mct 160)
- [x] Implement backprop
- [ ] Opt: Implement a tree policy function in mcts node_to_leaf
- [x] Opt: Dont choose argmax, use prob distribution (anet)
- [x] Reset board after game is over
- [x] Fortsette refactoring (reformat)
- [x] Not allowed to overwrite another players piece on the board
- [x] Add player and position as parameters to find_neighbours function

## Questions to studass
- [ ] TIPS for: Learning rate, Nodes in hidden layers, Activation func, optimizer
- [x] PLS HELP: model.fit does not work in train_model
- [x] Set new root and discrad everything else? Mener de slett treet over og sett ny subtree som tree slik at new root har none parents?
- [x] How to control explore and not?
- [x] How does the network know which moves are legal? - vet ikke, men vi sender inn null på moves som ikke er lov sånn at det ikke blir valgt
- [x] What should target be in train model?
- [x] How to represent a state in MCT
- [x] Does reformat board work with np.where(Piece.player == 1) ?
- [x] Should we rollout from all children, or follow tree-policy in choosing this as well? - Only one of the children
- [x] How should we represent board for input to anet?

- [x] SparseCategorialCrossentropy or CategoricalCrossentropy? Should we use one-hot encoding?
- [x] How to know when game is over? A*? Keep track of neighbours?

- [x] Can we use same initializer for weights and biases in neural net
- [x] What values should we use in the initializer
- [x] Should we initialize weights and biases in input/output layer as well?
- [x] Should we use MeanSquaredError as loss function? -> cross-entropy

## Tips from studass
- State: add one for which player makes the move
- Loss: use cross-entropy - this is also stated in the project paper
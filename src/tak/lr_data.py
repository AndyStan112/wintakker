import random
from tak.core import Result, WHITE, BLACK, TakState
from tak.ai import TakSearch, TakEvaluator, TakFeatures, TakWeights
from tqdm import tqdm


def play_game(search, size=5):
    states = []
    state = TakState.from_size(size)

    while True:
        result = state.compute_result(1 - state.to_move)
        if result != Result.ONGOING:
            break

        feats = search.evaluator.f.extract(state, state.to_move)
        states.append((feats, state.to_move))

        move = search.choose_move(state)
        state = state.clone()
        search._apply(state, move)

    if result == Result.DRAW:
        outcome = 0
    elif result == Result.WHITE_WIN:
        outcome = 1
    else:
        outcome = -1

    data = []
    for feats, player in states:
        y = outcome if player == WHITE else -outcome
        data.append((feats, y))

    return data


def generate_dataset(games=200, size=5):
    features = TakFeatures()
    weights = TakWeights()
    evaluator = TakEvaluator(weights, features)
    search = TakSearch(evaluator=evaluator, max_depth=1)

    dataset = []

    for _ in tqdm(range(games)):
        dataset.extend(play_game(search, size))

    return dataset

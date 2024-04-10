import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/double_hist/trainlength10/s1/double_hist_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 4
        elif q_position in {1, 3}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 1

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {8, 2, 11}:
            return k_position == 3
        elif q_position in {3, 6, 10, 12, 13, 14, 15, 17, 18}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {16, 9, 5}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 17

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 3, 6, 8, 18}:
            return k_position == 6
        elif q_position in {19, 2, 11}:
            return k_position == 5
        elif q_position in {4, 5}:
            return k_position == 2
        elif q_position in {9, 7}:
            return k_position == 1
        elif q_position in {16, 10}:
            return k_position == 18
        elif q_position in {17, 12, 13, 14}:
            return k_position == 16
        elif q_position in {15}:
            return k_position == 10

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 3, 4, 5, 8, 9, 13, 17}:
            return k_position == 6
        elif q_position in {2, 19, 15, 7}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 17
        elif q_position in {11, 12, 14, 16, 18}:
            return k_position == 7

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"0", "2", "4", "5", "3", "1"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 9

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0", "2", "4", "5", "3", "1"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 13

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "<pad>"
        elif q_token in {"2", "4", "<s>", "5", "3", "1"}:
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        if key in {
            (1, 2),
            (1, 8),
            (2, 2),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 6),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (10, 2),
            (11, 2),
            (12, 2),
            (13, 2),
            (14, 2),
            (15, 2),
            (16, 2),
            (17, 2),
            (18, 2),
            (19, 2),
        }:
            return 8
        elif key in {
            (0, 9),
            (1, 9),
            (2, 9),
            (3, 9),
            (10, 9),
            (11, 9),
            (12, 9),
            (13, 9),
            (14, 9),
            (15, 9),
            (16, 9),
            (17, 9),
            (18, 9),
            (19, 9),
        }:
            return 17
        return 2

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 4),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 15),
            (0, 16),
            (1, 0),
            (1, 1),
            (1, 4),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (2, 0),
            (2, 1),
            (2, 4),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (10, 0),
            (10, 1),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 4),
            (12, 0),
            (12, 4),
            (13, 0),
            (13, 4),
            (14, 0),
            (14, 4),
            (15, 0),
            (15, 4),
            (16, 0),
            (16, 4),
            (17, 0),
            (17, 4),
            (18, 0),
            (18, 4),
            (19, 0),
            (19, 4),
        }:
            return 4
        return 18

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one, num_attn_0_1_output):
        key = (one, num_attn_0_1_output)
        return 2

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(ones, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_3_output, position):
        if attn_0_3_output in {0, 1}:
            return position == 7
        elif attn_0_3_output in {2, 4}:
            return position == 5
        elif attn_0_3_output in {3}:
            return position == 9
        elif attn_0_3_output in {5, 10, 11, 12, 13, 14, 15, 17, 18, 19}:
            return position == 6
        elif attn_0_3_output in {6}:
            return position == 1
        elif attn_0_3_output in {7}:
            return position == 4
        elif attn_0_3_output in {8}:
            return position == 0
        elif attn_0_3_output in {9}:
            return position == 8
        elif attn_0_3_output in {16}:
            return position == 11

    attn_1_0_pattern = select_closest(positions, attn_0_3_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, attn_0_1_output):
        if token in {"0"}:
            return attn_0_1_output == 2
        elif token in {"4", "5", "1"}:
            return attn_0_1_output == 7
        elif token in {"2"}:
            return attn_0_1_output == 4
        elif token in {"3"}:
            return attn_0_1_output == 8
        elif token in {"<s>"}:
            return attn_0_1_output == 5

    attn_1_1_pattern = select_closest(attn_0_1_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {0, 4, 13}:
            return attn_0_2_output == 5
        elif attn_0_3_output in {1}:
            return attn_0_2_output == 7
        elif attn_0_3_output in {2}:
            return attn_0_2_output == 13
        elif attn_0_3_output in {3}:
            return attn_0_2_output == 9
        elif attn_0_3_output in {5, 8, 12, 16, 17}:
            return attn_0_2_output == 6
        elif attn_0_3_output in {18, 6, 14}:
            return attn_0_2_output == 2
        elif attn_0_3_output in {19, 7}:
            return attn_0_2_output == 1
        elif attn_0_3_output in {9}:
            return attn_0_2_output == 15
        elif attn_0_3_output in {10}:
            return attn_0_2_output == 18
        elif attn_0_3_output in {11}:
            return attn_0_2_output == 10
        elif attn_0_3_output in {15}:
            return attn_0_2_output == 8

    attn_1_2_pattern = select_closest(attn_0_2_outputs, attn_0_3_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, attn_0_0_output):
        if attn_0_3_output in {0, 16, 2, 3}:
            return attn_0_0_output == 9
        elif attn_0_3_output in {1, 6}:
            return attn_0_0_output == 7
        elif attn_0_3_output in {11, 4}:
            return attn_0_0_output == 5
        elif attn_0_3_output in {5}:
            return attn_0_0_output == 3
        elif attn_0_3_output in {7}:
            return attn_0_0_output == 1
        elif attn_0_3_output in {8}:
            return attn_0_0_output == 11
        elif attn_0_3_output in {9, 12}:
            return attn_0_0_output == 13
        elif attn_0_3_output in {10, 13, 14, 17, 19}:
            return attn_0_0_output == 6
        elif attn_0_3_output in {15}:
            return attn_0_0_output == 15
        elif attn_0_3_output in {18}:
            return attn_0_0_output == 8

    attn_1_3_pattern = select_closest(attn_0_0_outputs, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, position):
        if attn_0_2_output in {0, 12, 13, 18, 19}:
            return position == 4
        elif attn_0_2_output in {1, 2, 3, 14, 15, 17}:
            return position == 1
        elif attn_0_2_output in {4, 8, 9, 10, 16}:
            return position == 3
        elif attn_0_2_output in {5, 6, 7}:
            return position == 5
        elif attn_0_2_output in {11}:
            return position == 2

    num_attn_1_0_pattern = select(positions, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0}:
            return num_mlp_0_0_output == 1
        elif attn_0_0_output in {1, 11, 13}:
            return num_mlp_0_0_output == 2
        elif attn_0_0_output in {2, 18}:
            return num_mlp_0_0_output == 9
        elif attn_0_0_output in {3}:
            return num_mlp_0_0_output == 8
        elif attn_0_0_output in {17, 4}:
            return num_mlp_0_0_output == 5
        elif attn_0_0_output in {5, 14}:
            return num_mlp_0_0_output == 6
        elif attn_0_0_output in {19, 12, 6}:
            return num_mlp_0_0_output == 4
        elif attn_0_0_output in {16, 9, 7}:
            return num_mlp_0_0_output == 10
        elif attn_0_0_output in {8, 15}:
            return num_mlp_0_0_output == 11
        elif attn_0_0_output in {10}:
            return num_mlp_0_0_output == 16

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 16
        elif attn_0_2_output in {16, 1}:
            return position == 19
        elif attn_0_2_output in {2, 3, 4, 5, 6, 7, 14}:
            return position == 14
        elif attn_0_2_output in {8, 17, 11, 12}:
            return position == 11
        elif attn_0_2_output in {9}:
            return position == 12
        elif attn_0_2_output in {10, 19}:
            return position == 13
        elif attn_0_2_output in {13}:
            return position == 17
        elif attn_0_2_output in {15}:
            return position == 10
        elif attn_0_2_output in {18}:
            return position == 9

    num_attn_1_2_pattern = select(positions, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, attn_0_0_output):
        if attn_0_2_output in {0, 1, 9}:
            return attn_0_0_output == 13
        elif attn_0_2_output in {2, 11}:
            return attn_0_0_output == 15
        elif attn_0_2_output in {3}:
            return attn_0_0_output == 18
        elif attn_0_2_output in {4}:
            return attn_0_0_output == 17
        elif attn_0_2_output in {13, 5}:
            return attn_0_0_output == 11
        elif attn_0_2_output in {6, 7}:
            return attn_0_0_output == 9
        elif attn_0_2_output in {8}:
            return attn_0_0_output == 3
        elif attn_0_2_output in {10, 12}:
            return attn_0_0_output == 14
        elif attn_0_2_output in {14}:
            return attn_0_0_output == 16
        elif attn_0_2_output in {16, 17, 15}:
            return attn_0_0_output == 12
        elif attn_0_2_output in {18}:
            return attn_0_0_output == 8
        elif attn_0_2_output in {19}:
            return attn_0_0_output == 7

    num_attn_1_3_pattern = select(attn_0_0_outputs, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output):
        key = attn_0_0_output
        if key in {15}:
            return 4
        return 8

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_0_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 11),
            (0, 12),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 18),
            (0, 19),
            (1, 12),
            (1, 19),
            (3, 12),
            (3, 19),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 11),
            (4, 12),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 5),
            (5, 11),
            (5, 12),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 5),
            (12, 11),
            (12, 12),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 5),
            (13, 11),
            (13, 12),
            (13, 14),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (14, 12),
            (14, 19),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (16, 0),
            (16, 2),
            (16, 3),
            (16, 5),
            (16, 11),
            (16, 12),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 18),
            (16, 19),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 5),
            (17, 11),
            (17, 12),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 5),
            (18, 11),
            (18, 12),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 5),
            (19, 11),
            (19, 12),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
        }:
            return 0
        return 11

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        return 14

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(num_mlp_1_0_output, attn_0_3_output):
        if num_mlp_1_0_output in {0, 8}:
            return attn_0_3_output == 5
        elif num_mlp_1_0_output in {1, 9, 11, 15, 17}:
            return attn_0_3_output == 6
        elif num_mlp_1_0_output in {2, 14}:
            return attn_0_3_output == 1
        elif num_mlp_1_0_output in {3}:
            return attn_0_3_output == 17
        elif num_mlp_1_0_output in {4}:
            return attn_0_3_output == 15
        elif num_mlp_1_0_output in {5}:
            return attn_0_3_output == 0
        elif num_mlp_1_0_output in {13, 6}:
            return attn_0_3_output == 4
        elif num_mlp_1_0_output in {7}:
            return attn_0_3_output == 18
        elif num_mlp_1_0_output in {16, 10}:
            return attn_0_3_output == 3
        elif num_mlp_1_0_output in {12}:
            return attn_0_3_output == 9
        elif num_mlp_1_0_output in {18}:
            return attn_0_3_output == 16
        elif num_mlp_1_0_output in {19}:
            return attn_0_3_output == 10

    attn_2_0_pattern = select_closest(
        attn_0_3_outputs, num_mlp_1_0_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_0_output, token):
        if attn_0_0_output in {0, 1, 2, 3, 4, 8, 9, 10, 12, 14, 15, 18}:
            return token == ""
        elif attn_0_0_output in {5}:
            return token == "3"
        elif attn_0_0_output in {6}:
            return token == "2"
        elif attn_0_0_output in {7}:
            return token == "1"
        elif attn_0_0_output in {16, 11, 13}:
            return token == "<s>"
        elif attn_0_0_output in {17}:
            return token == "5"
        elif attn_0_0_output in {19}:
            return token == "4"

    attn_2_1_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_0_output, position):
        if attn_0_0_output in {0, 2, 4, 6, 17, 18}:
            return position == 7
        elif attn_0_0_output in {1}:
            return position == 9
        elif attn_0_0_output in {3}:
            return position == 13
        elif attn_0_0_output in {16, 5}:
            return position == 17
        elif attn_0_0_output in {7}:
            return position == 3
        elif attn_0_0_output in {8}:
            return position == 4
        elif attn_0_0_output in {9}:
            return position == 2
        elif attn_0_0_output in {10, 11, 13, 14, 15, 19}:
            return position == 6
        elif attn_0_0_output in {12}:
            return position == 5

    attn_2_2_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 8
        elif q_position in {8, 1, 4, 5}:
            return k_position == 9
        elif q_position in {9, 2, 6}:
            return k_position == 4
        elif q_position in {3}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 0
        elif q_position in {10, 12, 13, 15, 17, 18, 19}:
            return k_position == 6
        elif q_position in {11, 14}:
            return k_position == 7
        elif q_position in {16}:
            return k_position == 5

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_1_1_output, attn_1_0_output):
        if mlp_1_1_output in {0, 3, 5, 12, 19}:
            return attn_1_0_output == 11
        elif mlp_1_1_output in {1, 4}:
            return attn_1_0_output == 16
        elif mlp_1_1_output in {2, 18}:
            return attn_1_0_output == 18
        elif mlp_1_1_output in {8, 17, 6}:
            return attn_1_0_output == 13
        elif mlp_1_1_output in {16, 7}:
            return attn_1_0_output == 10
        elif mlp_1_1_output in {9}:
            return attn_1_0_output == 12
        elif mlp_1_1_output in {10, 15}:
            return attn_1_0_output == 19
        elif mlp_1_1_output in {11}:
            return attn_1_0_output == 14
        elif mlp_1_1_output in {13}:
            return attn_1_0_output == 17
        elif mlp_1_1_output in {14}:
            return attn_1_0_output == 3

    num_attn_2_0_pattern = select(attn_1_0_outputs, mlp_1_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, position):
        if attn_1_1_output in {0, 16}:
            return position == 1
        elif attn_1_1_output in {1, 4, 5, 12, 13, 18}:
            return position == 2
        elif attn_1_1_output in {19, 2, 10, 15}:
            return position == 4
        elif attn_1_1_output in {17, 3, 14}:
            return position == 3
        elif attn_1_1_output in {6}:
            return position == 6
        elif attn_1_1_output in {8, 7}:
            return position == 7
        elif attn_1_1_output in {9}:
            return position == 8
        elif attn_1_1_output in {11}:
            return position == 0

    num_attn_2_1_pattern = select(positions, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, position):
        if attn_1_0_output in {0, 1, 18, 17}:
            return position == 1
        elif attn_1_0_output in {11, 2, 3, 4}:
            return position == 4
        elif attn_1_0_output in {5}:
            return position == 5
        elif attn_1_0_output in {6}:
            return position == 6
        elif attn_1_0_output in {15, 13, 7}:
            return position == 2
        elif attn_1_0_output in {8, 12}:
            return position == 17
        elif attn_1_0_output in {9}:
            return position == 14
        elif attn_1_0_output in {10}:
            return position == 11
        elif attn_1_0_output in {16, 14}:
            return position == 3
        elif attn_1_0_output in {19}:
            return position == 15

    num_attn_2_2_pattern = select(positions, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_3_output, attn_1_2_output):
        if attn_1_3_output in {0, 1, 14, 17, 18, 19}:
            return attn_1_2_output == 0
        elif attn_1_3_output in {2}:
            return attn_1_2_output == 3
        elif attn_1_3_output in {10, 3, 4, 13}:
            return attn_1_2_output == 4
        elif attn_1_3_output in {12, 5, 15}:
            return attn_1_2_output == 5
        elif attn_1_3_output in {6}:
            return attn_1_2_output == 18
        elif attn_1_3_output in {7}:
            return attn_1_2_output == 11
        elif attn_1_3_output in {8}:
            return attn_1_2_output == 8
        elif attn_1_3_output in {9}:
            return attn_1_2_output == 15
        elif attn_1_3_output in {11}:
            return attn_1_2_output == 14
        elif attn_1_3_output in {16}:
            return attn_1_2_output == 2

    num_attn_2_3_pattern = select(attn_1_2_outputs, attn_1_3_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_1_output, mlp_1_0_output):
        key = (num_mlp_0_1_output, mlp_1_0_output)
        return 5

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_1_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_1_output, attn_1_3_output):
        key = (mlp_1_1_output, attn_1_3_output)
        return 17

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_1_3_output):
        key = (num_attn_2_0_output, num_attn_1_3_output)
        if key in {
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (6, 7),
            (6, 8),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 11),
            (7, 12),
            (7, 13),
            (7, 14),
            (7, 15),
            (7, 16),
            (7, 17),
            (7, 18),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (9, 22),
            (9, 23),
            (9, 24),
            (9, 25),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 16),
            (11, 17),
            (11, 18),
            (11, 19),
            (11, 20),
            (11, 21),
            (11, 22),
            (11, 23),
            (11, 24),
            (11, 25),
            (11, 26),
            (11, 27),
            (11, 28),
            (11, 29),
            (11, 30),
            (11, 31),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
            (11, 47),
            (11, 48),
            (11, 49),
            (11, 50),
            (11, 51),
            (11, 52),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 26),
            (12, 27),
            (12, 28),
            (12, 29),
            (12, 30),
            (12, 31),
            (12, 32),
            (12, 33),
            (12, 34),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (12, 48),
            (12, 49),
            (12, 50),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (13, 15),
            (13, 16),
            (13, 17),
            (13, 18),
            (13, 19),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (13, 24),
            (13, 25),
            (13, 26),
            (13, 27),
            (13, 28),
            (13, 29),
            (13, 30),
            (13, 31),
            (13, 32),
            (13, 33),
            (13, 34),
            (13, 35),
            (13, 36),
            (13, 37),
            (13, 38),
            (13, 39),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (13, 50),
            (13, 51),
            (13, 52),
            (13, 53),
            (13, 54),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 25),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 29),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 36),
            (14, 37),
            (14, 38),
            (14, 39),
            (14, 40),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (14, 48),
            (14, 49),
            (14, 50),
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
            (14, 59),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 36),
            (15, 37),
            (15, 38),
            (15, 39),
            (15, 40),
            (15, 41),
            (15, 42),
            (15, 43),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (15, 50),
            (15, 51),
            (15, 52),
            (15, 53),
            (15, 54),
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 30),
            (16, 31),
            (16, 32),
            (16, 33),
            (16, 34),
            (16, 35),
            (16, 36),
            (16, 37),
            (16, 38),
            (16, 39),
            (16, 40),
            (16, 41),
            (16, 42),
            (16, 43),
            (16, 44),
            (16, 45),
            (16, 46),
            (16, 47),
            (16, 48),
            (16, 49),
            (16, 50),
            (16, 51),
            (16, 52),
            (16, 53),
            (16, 54),
            (16, 55),
            (16, 56),
            (16, 57),
            (16, 58),
            (16, 59),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 24),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (17, 30),
            (17, 31),
            (17, 32),
            (17, 33),
            (17, 34),
            (17, 35),
            (17, 36),
            (17, 37),
            (17, 38),
            (17, 39),
            (17, 40),
            (17, 41),
            (17, 42),
            (17, 43),
            (17, 44),
            (17, 45),
            (17, 46),
            (17, 47),
            (17, 48),
            (17, 49),
            (17, 50),
            (17, 51),
            (17, 52),
            (17, 53),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 57),
            (17, 58),
            (17, 59),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 29),
            (18, 30),
            (18, 31),
            (18, 32),
            (18, 33),
            (18, 34),
            (18, 35),
            (18, 36),
            (18, 37),
            (18, 38),
            (18, 39),
            (18, 40),
            (18, 41),
            (18, 42),
            (18, 43),
            (18, 44),
            (18, 45),
            (18, 46),
            (18, 47),
            (18, 48),
            (18, 49),
            (18, 50),
            (18, 51),
            (18, 52),
            (18, 53),
            (18, 54),
            (18, 55),
            (18, 56),
            (18, 57),
            (18, 58),
            (18, 59),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 28),
            (19, 29),
            (19, 30),
            (19, 31),
            (19, 32),
            (19, 33),
            (19, 34),
            (19, 35),
            (19, 36),
            (19, 37),
            (19, 38),
            (19, 39),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 43),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (19, 48),
            (19, 49),
            (19, 50),
            (19, 51),
            (19, 52),
            (19, 53),
            (19, 54),
            (19, 55),
            (19, 56),
            (19, 57),
            (19, 58),
            (19, 59),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (20, 26),
            (20, 27),
            (20, 28),
            (20, 29),
            (20, 30),
            (20, 31),
            (20, 32),
            (20, 33),
            (20, 34),
            (20, 35),
            (20, 36),
            (20, 37),
            (20, 38),
            (20, 39),
            (20, 40),
            (20, 41),
            (20, 42),
            (20, 43),
            (20, 44),
            (20, 45),
            (20, 46),
            (20, 47),
            (20, 48),
            (20, 49),
            (20, 50),
            (20, 51),
            (20, 52),
            (20, 53),
            (20, 54),
            (20, 55),
            (20, 56),
            (20, 57),
            (20, 58),
            (20, 59),
            (21, 23),
            (21, 24),
            (21, 25),
            (21, 26),
            (21, 27),
            (21, 28),
            (21, 29),
            (21, 30),
            (21, 31),
            (21, 32),
            (21, 33),
            (21, 34),
            (21, 35),
            (21, 36),
            (21, 37),
            (21, 38),
            (21, 39),
            (21, 40),
            (21, 41),
            (21, 42),
            (21, 43),
            (21, 44),
            (21, 45),
            (21, 46),
            (21, 47),
            (21, 48),
            (21, 49),
            (21, 50),
            (21, 51),
            (21, 52),
            (21, 53),
            (21, 54),
            (21, 55),
            (21, 56),
            (21, 57),
            (21, 58),
            (21, 59),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 31),
            (22, 32),
            (22, 33),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 43),
            (22, 44),
            (22, 45),
            (22, 46),
            (22, 47),
            (22, 48),
            (22, 49),
            (22, 50),
            (22, 51),
            (22, 52),
            (22, 53),
            (22, 54),
            (22, 55),
            (22, 56),
            (22, 57),
            (22, 58),
            (22, 59),
            (23, 25),
            (23, 26),
            (23, 27),
            (23, 28),
            (23, 29),
            (23, 30),
            (23, 31),
            (23, 32),
            (23, 33),
            (23, 34),
            (23, 35),
            (23, 36),
            (23, 37),
            (23, 38),
            (23, 39),
            (23, 40),
            (23, 41),
            (23, 42),
            (23, 43),
            (23, 44),
            (23, 45),
            (23, 46),
            (23, 47),
            (23, 48),
            (23, 49),
            (23, 50),
            (23, 51),
            (23, 52),
            (23, 53),
            (23, 54),
            (23, 55),
            (23, 56),
            (23, 57),
            (23, 58),
            (23, 59),
            (24, 26),
            (24, 27),
            (24, 28),
            (24, 29),
            (24, 30),
            (24, 31),
            (24, 32),
            (24, 33),
            (24, 34),
            (24, 35),
            (24, 36),
            (24, 37),
            (24, 38),
            (24, 39),
            (24, 40),
            (24, 41),
            (24, 42),
            (24, 43),
            (24, 44),
            (24, 45),
            (24, 46),
            (24, 47),
            (24, 48),
            (24, 49),
            (24, 50),
            (24, 51),
            (24, 52),
            (24, 53),
            (24, 54),
            (24, 55),
            (24, 56),
            (24, 57),
            (24, 58),
            (24, 59),
            (25, 27),
            (25, 28),
            (25, 29),
            (25, 30),
            (25, 31),
            (25, 32),
            (25, 33),
            (25, 34),
            (25, 35),
            (25, 36),
            (25, 37),
            (25, 38),
            (25, 39),
            (25, 40),
            (25, 41),
            (25, 42),
            (25, 43),
            (25, 44),
            (25, 45),
            (25, 46),
            (25, 47),
            (25, 48),
            (25, 49),
            (25, 50),
            (25, 51),
            (25, 52),
            (25, 53),
            (25, 54),
            (25, 55),
            (25, 56),
            (25, 57),
            (25, 58),
            (25, 59),
            (26, 28),
            (26, 29),
            (26, 30),
            (26, 31),
            (26, 32),
            (26, 33),
            (26, 34),
            (26, 35),
            (26, 36),
            (26, 37),
            (26, 38),
            (26, 39),
            (26, 40),
            (26, 41),
            (26, 42),
            (26, 43),
            (26, 44),
            (26, 45),
            (26, 46),
            (26, 47),
            (26, 48),
            (26, 49),
            (26, 50),
            (26, 51),
            (26, 52),
            (26, 53),
            (26, 54),
            (26, 55),
            (26, 56),
            (26, 57),
            (26, 58),
            (26, 59),
            (27, 29),
            (27, 30),
            (27, 31),
            (27, 32),
            (27, 33),
            (27, 34),
            (27, 35),
            (27, 36),
            (27, 37),
            (27, 38),
            (27, 39),
            (27, 40),
            (27, 41),
            (27, 42),
            (27, 43),
            (27, 44),
            (27, 45),
            (27, 46),
            (27, 47),
            (27, 48),
            (27, 49),
            (27, 50),
            (27, 51),
            (27, 52),
            (27, 53),
            (27, 54),
            (27, 55),
            (27, 56),
            (27, 57),
            (27, 58),
            (27, 59),
            (28, 30),
            (28, 31),
            (28, 32),
            (28, 33),
            (28, 34),
            (28, 35),
            (28, 36),
            (28, 37),
            (28, 38),
            (28, 39),
            (28, 40),
            (28, 41),
            (28, 42),
            (28, 43),
            (28, 44),
            (28, 45),
            (28, 46),
            (28, 47),
            (28, 48),
            (28, 49),
            (28, 50),
            (28, 51),
            (28, 52),
            (28, 53),
            (28, 54),
            (28, 55),
            (28, 56),
            (28, 57),
            (28, 58),
            (28, 59),
            (29, 31),
            (29, 32),
            (29, 33),
            (29, 34),
            (29, 35),
            (29, 36),
            (29, 37),
            (29, 38),
            (29, 39),
            (29, 40),
            (29, 41),
            (29, 42),
            (29, 43),
            (29, 44),
            (29, 45),
            (29, 46),
            (29, 47),
            (29, 48),
            (29, 49),
            (29, 50),
            (29, 51),
            (29, 52),
            (29, 53),
            (29, 54),
            (29, 55),
            (29, 56),
            (29, 57),
            (29, 58),
            (29, 59),
            (30, 32),
            (30, 33),
            (30, 34),
            (30, 35),
            (30, 36),
            (30, 37),
            (30, 38),
            (30, 39),
            (30, 40),
            (30, 41),
            (30, 42),
            (30, 43),
            (30, 44),
            (30, 45),
            (30, 46),
            (30, 47),
            (30, 48),
            (30, 49),
            (30, 50),
            (30, 51),
            (30, 52),
            (30, 53),
            (30, 54),
            (30, 55),
            (30, 56),
            (30, 57),
            (30, 58),
            (30, 59),
            (31, 33),
            (31, 34),
            (31, 35),
            (31, 36),
            (31, 37),
            (31, 38),
            (31, 39),
            (31, 40),
            (31, 41),
            (31, 42),
            (31, 43),
            (31, 44),
            (31, 45),
            (31, 46),
            (31, 47),
            (31, 48),
            (31, 49),
            (31, 50),
            (31, 51),
            (31, 52),
            (31, 53),
            (31, 54),
            (31, 55),
            (31, 56),
            (31, 57),
            (31, 58),
            (31, 59),
            (32, 34),
            (32, 35),
            (32, 36),
            (32, 37),
            (32, 38),
            (32, 39),
            (32, 40),
            (32, 41),
            (32, 42),
            (32, 43),
            (32, 44),
            (32, 45),
            (32, 46),
            (32, 47),
            (32, 48),
            (32, 49),
            (32, 50),
            (32, 51),
            (32, 52),
            (32, 53),
            (32, 54),
            (32, 55),
            (32, 56),
            (32, 57),
            (32, 58),
            (32, 59),
            (33, 35),
            (33, 36),
            (33, 37),
            (33, 38),
            (33, 39),
            (33, 40),
            (33, 41),
            (33, 42),
            (33, 43),
            (33, 44),
            (33, 45),
            (33, 46),
            (33, 47),
            (33, 48),
            (33, 49),
            (33, 50),
            (33, 51),
            (33, 52),
            (33, 53),
            (33, 54),
            (33, 55),
            (33, 56),
            (33, 57),
            (33, 58),
            (33, 59),
            (34, 36),
            (34, 37),
            (34, 38),
            (34, 39),
            (34, 40),
            (34, 41),
            (34, 42),
            (34, 43),
            (34, 44),
            (34, 45),
            (34, 46),
            (34, 47),
            (34, 48),
            (34, 49),
            (34, 50),
            (34, 51),
            (34, 52),
            (34, 53),
            (34, 54),
            (34, 55),
            (34, 56),
            (34, 57),
            (34, 58),
            (34, 59),
            (35, 38),
            (35, 39),
            (35, 40),
            (35, 41),
            (35, 42),
            (35, 43),
            (35, 44),
            (35, 45),
            (35, 46),
            (35, 47),
            (35, 48),
            (35, 49),
            (35, 50),
            (35, 51),
            (35, 52),
            (35, 53),
            (35, 54),
            (35, 55),
            (35, 56),
            (35, 57),
            (35, 58),
            (35, 59),
            (36, 39),
            (36, 40),
            (36, 41),
            (36, 42),
            (36, 43),
            (36, 44),
            (36, 45),
            (36, 46),
            (36, 47),
            (36, 48),
            (36, 49),
            (36, 50),
            (36, 51),
            (36, 52),
            (36, 53),
            (36, 54),
            (36, 55),
            (36, 56),
            (36, 57),
            (36, 58),
            (36, 59),
            (37, 40),
            (37, 41),
            (37, 42),
            (37, 43),
            (37, 44),
            (37, 45),
            (37, 46),
            (37, 47),
            (37, 48),
            (37, 49),
            (37, 50),
            (37, 51),
            (37, 52),
            (37, 53),
            (37, 54),
            (37, 55),
            (37, 56),
            (37, 57),
            (37, 58),
            (37, 59),
            (38, 41),
            (38, 42),
            (38, 43),
            (38, 44),
            (38, 45),
            (38, 46),
            (38, 47),
            (38, 48),
            (38, 49),
            (38, 50),
            (38, 51),
            (38, 52),
            (38, 53),
            (38, 54),
            (38, 55),
            (38, 56),
            (38, 57),
            (38, 58),
            (38, 59),
            (39, 42),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 48),
            (39, 49),
            (39, 50),
            (39, 51),
            (39, 52),
            (39, 53),
            (39, 54),
            (39, 55),
            (39, 56),
            (39, 57),
            (39, 58),
            (39, 59),
            (40, 43),
            (40, 44),
            (40, 45),
            (40, 46),
            (40, 47),
            (40, 48),
            (40, 49),
            (40, 50),
            (40, 51),
            (40, 52),
            (40, 53),
            (40, 54),
            (40, 55),
            (40, 56),
            (40, 57),
            (40, 58),
            (40, 59),
            (41, 44),
            (41, 45),
            (41, 46),
            (41, 47),
            (41, 48),
            (41, 49),
            (41, 50),
            (41, 51),
            (41, 52),
            (41, 53),
            (41, 54),
            (41, 55),
            (41, 56),
            (41, 57),
            (41, 58),
            (41, 59),
            (42, 45),
            (42, 46),
            (42, 47),
            (42, 48),
            (42, 49),
            (42, 50),
            (42, 51),
            (42, 52),
            (42, 53),
            (42, 54),
            (42, 55),
            (42, 56),
            (42, 57),
            (42, 58),
            (42, 59),
            (43, 46),
            (43, 47),
            (43, 48),
            (43, 49),
            (43, 50),
            (43, 51),
            (43, 52),
            (43, 53),
            (43, 54),
            (43, 55),
            (43, 56),
            (43, 57),
            (43, 58),
            (43, 59),
            (44, 47),
            (44, 48),
            (44, 49),
            (44, 50),
            (44, 51),
            (44, 52),
            (44, 53),
            (44, 54),
            (44, 55),
            (44, 56),
            (44, 57),
            (44, 58),
            (44, 59),
            (45, 48),
            (45, 49),
            (45, 50),
            (45, 51),
            (45, 52),
            (45, 53),
            (45, 54),
            (45, 55),
            (45, 56),
            (45, 57),
            (45, 58),
            (45, 59),
            (46, 49),
            (46, 50),
            (46, 51),
            (46, 52),
            (46, 53),
            (46, 54),
            (46, 55),
            (46, 56),
            (46, 57),
            (46, 58),
            (46, 59),
            (47, 50),
            (47, 51),
            (47, 52),
            (47, 53),
            (47, 54),
            (47, 55),
            (47, 56),
            (47, 57),
            (47, 58),
            (47, 59),
            (48, 51),
            (48, 52),
            (48, 53),
            (48, 54),
            (48, 55),
            (48, 56),
            (48, 57),
            (48, 58),
            (48, 59),
            (49, 52),
            (49, 53),
            (49, 54),
            (49, 55),
            (49, 56),
            (49, 57),
            (49, 58),
            (49, 59),
            (50, 53),
            (50, 54),
            (50, 55),
            (50, 56),
            (50, 57),
            (50, 58),
            (50, 59),
            (51, 54),
            (51, 55),
            (51, 56),
            (51, 57),
            (51, 58),
            (51, 59),
            (52, 55),
            (52, 56),
            (52, 57),
            (52, 58),
            (52, 59),
            (53, 56),
            (53, 57),
            (53, 58),
            (53, 59),
            (54, 57),
            (54, 58),
            (54, 59),
            (55, 58),
            (55, 59),
            (56, 59),
        }:
            return 5
        return 17

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_2_1_output):
        key = (num_attn_1_0_output, num_attn_2_1_output)
        if key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 48),
            (0, 49),
            (0, 50),
            (0, 51),
            (0, 52),
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 48),
            (1, 49),
            (1, 50),
            (1, 51),
            (1, 52),
            (1, 53),
            (1, 54),
            (1, 55),
            (1, 56),
            (1, 57),
            (1, 58),
            (1, 59),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 48),
            (2, 49),
            (2, 50),
            (2, 51),
            (2, 52),
            (2, 53),
            (2, 54),
            (2, 55),
            (2, 56),
            (2, 57),
            (2, 58),
            (2, 59),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (3, 20),
            (3, 21),
            (3, 22),
            (3, 23),
            (3, 24),
            (3, 25),
            (3, 26),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
            (3, 32),
            (3, 33),
            (3, 34),
            (3, 35),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
            (3, 48),
            (3, 49),
            (3, 50),
            (3, 51),
            (3, 52),
            (3, 53),
            (3, 54),
            (3, 55),
            (3, 56),
            (3, 57),
            (3, 58),
            (3, 59),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (4, 16),
            (4, 17),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (4, 30),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 34),
            (4, 35),
            (4, 36),
            (4, 37),
            (4, 38),
            (4, 39),
            (4, 40),
            (4, 41),
            (4, 42),
            (4, 43),
            (4, 44),
            (4, 45),
            (4, 46),
            (4, 47),
            (4, 48),
            (4, 49),
            (4, 50),
            (4, 51),
            (4, 52),
            (4, 53),
            (4, 54),
            (4, 55),
            (4, 56),
            (4, 57),
            (4, 58),
            (4, 59),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 16),
            (5, 17),
            (5, 18),
            (5, 19),
            (5, 20),
            (5, 21),
            (5, 22),
            (5, 23),
            (5, 24),
            (5, 25),
            (5, 26),
            (5, 27),
            (5, 28),
            (5, 29),
            (5, 30),
            (5, 31),
            (5, 32),
            (5, 33),
            (5, 34),
            (5, 35),
            (5, 36),
            (5, 37),
            (5, 38),
            (5, 39),
            (5, 40),
            (5, 41),
            (5, 42),
            (5, 43),
            (5, 44),
            (5, 45),
            (5, 46),
            (5, 47),
            (5, 48),
            (5, 49),
            (5, 50),
            (5, 51),
            (5, 52),
            (5, 53),
            (5, 54),
            (5, 55),
            (5, 56),
            (5, 57),
            (5, 58),
            (5, 59),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (6, 50),
            (6, 51),
            (6, 52),
            (6, 53),
            (6, 54),
            (6, 55),
            (6, 56),
            (6, 57),
            (6, 58),
            (6, 59),
            (7, 19),
            (7, 20),
            (7, 21),
            (7, 22),
            (7, 23),
            (7, 24),
            (7, 25),
            (7, 26),
            (7, 27),
            (7, 28),
            (7, 29),
            (7, 30),
            (7, 31),
            (7, 32),
            (7, 33),
            (7, 34),
            (7, 35),
            (7, 36),
            (7, 37),
            (7, 38),
            (7, 39),
            (7, 40),
            (7, 41),
            (7, 42),
            (7, 43),
            (7, 44),
            (7, 45),
            (7, 46),
            (7, 47),
            (7, 48),
            (7, 49),
            (7, 50),
            (7, 51),
            (7, 52),
            (7, 53),
            (7, 54),
            (7, 55),
            (7, 56),
            (7, 57),
            (7, 58),
            (7, 59),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (8, 30),
            (8, 31),
            (8, 32),
            (8, 33),
            (8, 34),
            (8, 35),
            (8, 36),
            (8, 37),
            (8, 38),
            (8, 39),
            (8, 40),
            (8, 41),
            (8, 42),
            (8, 43),
            (8, 44),
            (8, 45),
            (8, 46),
            (8, 47),
            (8, 48),
            (8, 49),
            (8, 50),
            (8, 51),
            (8, 52),
            (8, 53),
            (8, 54),
            (8, 55),
            (8, 56),
            (8, 57),
            (8, 58),
            (8, 59),
            (9, 26),
            (9, 27),
            (9, 28),
            (9, 29),
            (9, 30),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 35),
            (9, 36),
            (9, 37),
            (9, 38),
            (9, 39),
            (9, 40),
            (9, 41),
            (9, 42),
            (9, 43),
            (9, 44),
            (9, 45),
            (9, 46),
            (9, 47),
            (9, 48),
            (9, 49),
            (9, 50),
            (9, 51),
            (9, 52),
            (9, 53),
            (9, 54),
            (9, 55),
            (9, 56),
            (9, 57),
            (9, 58),
            (9, 59),
            (10, 29),
            (10, 30),
            (10, 31),
            (10, 32),
            (10, 33),
            (10, 34),
            (10, 35),
            (10, 36),
            (10, 37),
            (10, 38),
            (10, 39),
            (10, 40),
            (10, 41),
            (10, 42),
            (10, 43),
            (10, 44),
            (10, 45),
            (10, 46),
            (10, 47),
            (10, 48),
            (10, 49),
            (10, 50),
            (10, 51),
            (10, 52),
            (10, 53),
            (10, 54),
            (10, 55),
            (10, 56),
            (10, 57),
            (10, 58),
            (10, 59),
            (11, 32),
            (11, 33),
            (11, 34),
            (11, 35),
            (11, 36),
            (11, 37),
            (11, 38),
            (11, 39),
            (11, 40),
            (11, 41),
            (11, 42),
            (11, 43),
            (11, 44),
            (11, 45),
            (11, 46),
            (11, 47),
            (11, 48),
            (11, 49),
            (11, 50),
            (11, 51),
            (11, 52),
            (11, 53),
            (11, 54),
            (11, 55),
            (11, 56),
            (11, 57),
            (11, 58),
            (11, 59),
            (12, 35),
            (12, 36),
            (12, 37),
            (12, 38),
            (12, 39),
            (12, 40),
            (12, 41),
            (12, 42),
            (12, 43),
            (12, 44),
            (12, 45),
            (12, 46),
            (12, 47),
            (12, 48),
            (12, 49),
            (12, 50),
            (12, 51),
            (12, 52),
            (12, 53),
            (12, 54),
            (12, 55),
            (12, 56),
            (12, 57),
            (12, 58),
            (12, 59),
            (13, 38),
            (13, 39),
            (13, 40),
            (13, 41),
            (13, 42),
            (13, 43),
            (13, 44),
            (13, 45),
            (13, 46),
            (13, 47),
            (13, 48),
            (13, 49),
            (13, 50),
            (13, 51),
            (13, 52),
            (13, 53),
            (13, 54),
            (13, 55),
            (13, 56),
            (13, 57),
            (13, 58),
            (13, 59),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 47),
            (14, 48),
            (14, 49),
            (14, 50),
            (14, 51),
            (14, 52),
            (14, 53),
            (14, 54),
            (14, 55),
            (14, 56),
            (14, 57),
            (14, 58),
            (14, 59),
            (15, 44),
            (15, 45),
            (15, 46),
            (15, 47),
            (15, 48),
            (15, 49),
            (15, 50),
            (15, 51),
            (15, 52),
            (15, 53),
            (15, 54),
            (15, 55),
            (15, 56),
            (15, 57),
            (15, 58),
            (15, 59),
            (16, 47),
            (16, 48),
            (16, 49),
            (16, 50),
            (16, 51),
            (16, 52),
            (16, 53),
            (16, 54),
            (16, 55),
            (16, 56),
            (16, 57),
            (16, 58),
            (16, 59),
            (17, 50),
            (17, 51),
            (17, 52),
            (17, 53),
            (17, 54),
            (17, 55),
            (17, 56),
            (17, 57),
            (17, 58),
            (17, 59),
            (18, 53),
            (18, 54),
            (18, 55),
            (18, 56),
            (18, 57),
            (18, 58),
            (18, 59),
            (19, 56),
            (19, 57),
            (19, 58),
            (19, 59),
        }:
            return 4
        elif key in {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}:
            return 10
        return 7

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "3", "4", "0", "1", "3", "5"]))

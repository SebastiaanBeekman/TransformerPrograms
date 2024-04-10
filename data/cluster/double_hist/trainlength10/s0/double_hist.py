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
        "output/length/rasp/double_hist/trainlength10/s0/double_hist_weights.csv",
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
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 3, 10, 11, 12, 13, 14, 16, 17, 18}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {8, 5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 14
        elif q_position in {7}:
            return k_position == 0
        elif q_position in {9, 15}:
            return k_position == 5
        elif q_position in {19}:
            return k_position == 8

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 4
        elif q_position in {1, 18, 11, 17}:
            return k_position == 5
        elif q_position in {2, 3, 5, 7, 8, 9, 10, 12, 13, 15, 16}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {14}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 9

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 7
        elif q_position in {1, 4, 13, 9}:
            return k_position == 6
        elif q_position in {2, 10, 7}:
            return k_position == 5
        elif q_position in {18, 3, 14, 15}:
            return k_position == 4
        elif q_position in {5, 11, 12, 16, 19}:
            return k_position == 9
        elif q_position in {8}:
            return k_position == 2
        elif q_position in {17}:
            return k_position == 16

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 2, 5, 6}:
            return k_position == 7
        elif q_position in {17, 18, 3}:
            return k_position == 5
        elif q_position in {4, 7, 8, 9, 11, 12, 13, 14, 16, 19}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 16

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
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

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 3, 4, 6, 7}:
            return k_position == 9
        elif q_position in {5, 15}:
            return k_position == 11
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 17
        elif q_position in {10}:
            return k_position == 15
        elif q_position in {11}:
            return k_position == 7
        elif q_position in {12, 16, 17, 18, 19}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {14}:
            return k_position == 12

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 2, 3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {10, 19, 6, 15}:
            return k_position == 2
        elif q_position in {7}:
            return k_position == 19
        elif q_position in {8, 17}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {11, 12, 14, 16, 18}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 15

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
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

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        if key in {
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
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
        }:
            return 3
        return 0

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_0_output):
        key = (attn_0_1_output, attn_0_0_output)
        return 16

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 10

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 19

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {0}:
            return position == 7
        elif attn_0_0_output in {1}:
            return position == 8
        elif attn_0_0_output in {2, 3, 11, 12, 13, 14, 15, 16}:
            return position == 6
        elif attn_0_0_output in {8, 4}:
            return position == 4
        elif attn_0_0_output in {5, 7}:
            return position == 5
        elif attn_0_0_output in {6}:
            return position == 0
        elif attn_0_0_output in {9}:
            return position == 1
        elif attn_0_0_output in {10}:
            return position == 16
        elif attn_0_0_output in {17}:
            return position == 13
        elif attn_0_0_output in {18}:
            return position == 18
        elif attn_0_0_output in {19}:
            return position == 12

    attn_1_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, attn_0_1_output):
        if position in {0, 4, 7}:
            return attn_0_1_output == 8
        elif position in {8, 1, 3}:
            return attn_0_1_output == 5
        elif position in {2}:
            return attn_0_1_output == 17
        elif position in {5, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return attn_0_1_output == 6
        elif position in {6}:
            return attn_0_1_output == 7
        elif position in {9}:
            return attn_0_1_output == 4
        elif position in {10}:
            return attn_0_1_output == 12

    attn_1_1_pattern = select_closest(attn_0_1_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {0}:
            return attn_0_3_output == 5
        elif attn_0_1_output in {1, 7}:
            return attn_0_3_output == 8
        elif attn_0_1_output in {2}:
            return attn_0_3_output == 9
        elif attn_0_1_output in {3, 4, 6, 10, 12, 16}:
            return attn_0_3_output == 7
        elif attn_0_1_output in {13, 5}:
            return attn_0_3_output == 6
        elif attn_0_1_output in {8}:
            return attn_0_3_output == 2
        elif attn_0_1_output in {9}:
            return attn_0_3_output == 4
        elif attn_0_1_output in {11}:
            return attn_0_3_output == 14
        elif attn_0_1_output in {14, 15}:
            return attn_0_3_output == 19
        elif attn_0_1_output in {17, 19}:
            return attn_0_3_output == 12
        elif attn_0_1_output in {18}:
            return attn_0_3_output == 13

    attn_1_2_pattern = select_closest(attn_0_3_outputs, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {0}:
            return k_attn_0_0_output == 9
        elif q_attn_0_0_output in {1, 2, 3, 5, 7, 10, 11, 14, 15, 17, 19}:
            return k_attn_0_0_output == 6
        elif q_attn_0_0_output in {4}:
            return k_attn_0_0_output == 5
        elif q_attn_0_0_output in {6}:
            return k_attn_0_0_output == 8
        elif q_attn_0_0_output in {8}:
            return k_attn_0_0_output == 2
        elif q_attn_0_0_output in {9}:
            return k_attn_0_0_output == 14
        elif q_attn_0_0_output in {12}:
            return k_attn_0_0_output == 17
        elif q_attn_0_0_output in {13}:
            return k_attn_0_0_output == 10
        elif q_attn_0_0_output in {16}:
            return k_attn_0_0_output == 13
        elif q_attn_0_0_output in {18}:
            return k_attn_0_0_output == 16

    attn_1_3_pattern = select_closest(attn_0_0_outputs, attn_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {0}:
            return position == 3
        elif attn_0_0_output in {1, 5}:
            return position == 12
        elif attn_0_0_output in {2, 6}:
            return position == 9
        elif attn_0_0_output in {3}:
            return position == 13
        elif attn_0_0_output in {4, 12}:
            return position == 14
        elif attn_0_0_output in {7}:
            return position == 16
        elif attn_0_0_output in {8, 13, 14}:
            return position == 17
        elif attn_0_0_output in {9}:
            return position == 15
        elif attn_0_0_output in {10, 19}:
            return position == 0
        elif attn_0_0_output in {17, 11}:
            return position == 18
        elif attn_0_0_output in {15}:
            return position == 7
        elif attn_0_0_output in {16}:
            return position == 2
        elif attn_0_0_output in {18}:
            return position == 1

    num_attn_1_0_pattern = select(positions, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 17, 2, 15}:
            return position == 1
        elif num_mlp_0_1_output in {1, 11, 4, 6}:
            return position == 4
        elif num_mlp_0_1_output in {18, 19, 10, 3}:
            return position == 2
        elif num_mlp_0_1_output in {5}:
            return position == 5
        elif num_mlp_0_1_output in {8, 14, 7}:
            return position == 6
        elif num_mlp_0_1_output in {16, 9, 12, 13}:
            return position == 3

    num_attn_1_1_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0}:
            return mlp_0_0_output == 8
        elif num_mlp_0_0_output in {8, 1, 11, 14}:
            return mlp_0_0_output == 10
        elif num_mlp_0_0_output in {2, 12, 6, 15}:
            return mlp_0_0_output == 13
        elif num_mlp_0_0_output in {16, 17, 3}:
            return mlp_0_0_output == 9
        elif num_mlp_0_0_output in {10, 4, 7}:
            return mlp_0_0_output == 11
        elif num_mlp_0_0_output in {5}:
            return mlp_0_0_output == 12
        elif num_mlp_0_0_output in {9}:
            return mlp_0_0_output == 1
        elif num_mlp_0_0_output in {13}:
            return mlp_0_0_output == 4
        elif num_mlp_0_0_output in {18}:
            return mlp_0_0_output == 14
        elif num_mlp_0_0_output in {19}:
            return mlp_0_0_output == 15

    num_attn_1_2_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {0, 18, 4}:
            return position == 1
        elif attn_0_1_output in {1, 2, 14, 15, 17}:
            return position == 2
        elif attn_0_1_output in {11, 10, 3, 6}:
            return position == 4
        elif attn_0_1_output in {16, 13, 19, 5}:
            return position == 3
        elif attn_0_1_output in {12, 7}:
            return position == 5
        elif attn_0_1_output in {8}:
            return position == 6
        elif attn_0_1_output in {9}:
            return position == 9

    num_attn_1_3_pattern = select(positions, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_1_3_output):
        key = (attn_1_1_output, attn_1_3_output)
        if key in {
            (0, 1),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 17),
            (0, 19),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 17),
            (2, 19),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 17),
            (10, 19),
            (11, 1),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
            (11, 17),
            (11, 18),
            (11, 19),
            (12, 0),
            (12, 1),
            (12, 4),
            (12, 11),
            (12, 12),
            (12, 13),
            (12, 14),
            (12, 15),
            (12, 16),
            (12, 17),
            (12, 18),
            (12, 19),
            (13, 12),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 17),
            (13, 19),
            (14, 11),
            (14, 12),
            (14, 13),
            (14, 14),
            (14, 15),
            (14, 17),
            (14, 19),
            (15, 14),
            (16, 0),
            (16, 1),
            (16, 4),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (17, 0),
            (17, 1),
            (17, 4),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
            (17, 19),
            (18, 1),
            (18, 11),
            (18, 12),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 17),
            (18, 18),
            (18, 19),
            (19, 1),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 17),
            (19, 18),
            (19, 19),
        }:
            return 10
        elif key in {
            (0, 0),
            (0, 2),
            (0, 3),
            (0, 9),
            (0, 10),
            (0, 18),
            (1, 0),
            (1, 13),
            (2, 0),
            (2, 18),
            (11, 0),
            (11, 9),
            (12, 3),
            (12, 9),
            (13, 0),
            (13, 9),
            (13, 18),
            (14, 0),
            (14, 18),
            (15, 0),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 15),
            (15, 17),
            (15, 18),
            (16, 3),
            (16, 9),
            (17, 9),
            (18, 0),
            (18, 3),
            (18, 9),
            (19, 0),
            (19, 3),
            (19, 9),
        }:
            return 17
        return 4

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 10),
            (0, 12),
            (0, 13),
            (0, 15),
            (0, 18),
            (1, 10),
            (2, 10),
            (5, 10),
            (10, 0),
            (10, 10),
            (10, 12),
            (10, 13),
            (10, 18),
            (11, 10),
            (11, 12),
            (11, 18),
            (12, 0),
            (12, 10),
            (12, 12),
            (12, 13),
            (12, 15),
            (12, 18),
            (13, 10),
            (13, 12),
            (13, 13),
            (13, 18),
            (14, 10),
            (15, 0),
            (15, 10),
            (15, 12),
            (15, 13),
            (15, 18),
            (16, 0),
            (16, 1),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 13),
            (16, 15),
            (16, 17),
            (16, 18),
            (17, 0),
            (17, 10),
            (17, 12),
            (17, 13),
            (17, 18),
            (18, 0),
            (18, 10),
            (18, 12),
            (18, 13),
            (18, 15),
            (18, 17),
            (18, 18),
            (19, 10),
        }:
            return 6
        return 11

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_0_0_output):
        key = (num_attn_1_3_output, num_attn_0_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (9, 4),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (38, 0),
            (38, 1),
            (38, 2),
            (39, 0),
            (39, 1),
            (39, 2),
        }:
            return 3
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        if key in {(0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1)}:
            return 10
        elif key in {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)}:
            return 7
        return 5

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 2, 16, 17, 18, 19}:
            return k_position == 6
        elif q_position in {8, 3, 6, 7}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 16
        elif q_position in {5}:
            return k_position == 18
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 15

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_1_output, attn_0_2_output):
        if attn_1_1_output in {0, 17, 14}:
            return attn_0_2_output == 3
        elif attn_1_1_output in {1, 4}:
            return attn_0_2_output == 9
        elif attn_1_1_output in {2}:
            return attn_0_2_output == 1
        elif attn_1_1_output in {3, 7}:
            return attn_0_2_output == 2
        elif attn_1_1_output in {5}:
            return attn_0_2_output == 4
        elif attn_1_1_output in {6, 9, 10, 12, 13, 18}:
            return attn_0_2_output == 7
        elif attn_1_1_output in {8}:
            return attn_0_2_output == 10
        elif attn_1_1_output in {11}:
            return attn_0_2_output == 18
        elif attn_1_1_output in {15}:
            return attn_0_2_output == 5
        elif attn_1_1_output in {16}:
            return attn_0_2_output == 8
        elif attn_1_1_output in {19}:
            return attn_0_2_output == 11

    attn_2_1_pattern = select_closest(attn_0_2_outputs, attn_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(num_mlp_1_0_output, attn_1_3_output):
        if num_mlp_1_0_output in {0, 4}:
            return attn_1_3_output == 14
        elif num_mlp_1_0_output in {1}:
            return attn_1_3_output == 10
        elif num_mlp_1_0_output in {2, 5, 11, 13, 14, 15}:
            return attn_1_3_output == 6
        elif num_mlp_1_0_output in {3}:
            return attn_1_3_output == 4
        elif num_mlp_1_0_output in {6}:
            return attn_1_3_output == 12
        elif num_mlp_1_0_output in {7}:
            return attn_1_3_output == 2
        elif num_mlp_1_0_output in {8, 9, 18, 19}:
            return attn_1_3_output == 7
        elif num_mlp_1_0_output in {17, 10}:
            return attn_1_3_output == 9
        elif num_mlp_1_0_output in {12}:
            return attn_1_3_output == 17
        elif num_mlp_1_0_output in {16}:
            return attn_1_3_output == 11

    attn_2_2_pattern = select_closest(
        attn_1_3_outputs, num_mlp_1_0_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, positions)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(num_mlp_1_0_output, position):
        if num_mlp_1_0_output in {0, 1, 10, 7}:
            return position == 8
        elif num_mlp_1_0_output in {2, 13}:
            return position == 9
        elif num_mlp_1_0_output in {11, 19, 3, 5}:
            return position == 6
        elif num_mlp_1_0_output in {8, 4, 6}:
            return position == 7
        elif num_mlp_1_0_output in {9, 14}:
            return position == 3
        elif num_mlp_1_0_output in {12}:
            return position == 1
        elif num_mlp_1_0_output in {15}:
            return position == 15
        elif num_mlp_1_0_output in {16}:
            return position == 11
        elif num_mlp_1_0_output in {17}:
            return position == 18
        elif num_mlp_1_0_output in {18}:
            return position == 14

    attn_2_3_pattern = select_closest(positions, num_mlp_1_0_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, positions)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, position):
        if attn_1_3_output in {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 19}:
            return position == 8
        elif attn_1_3_output in {7}:
            return position == 3
        elif attn_1_3_output in {8}:
            return position == 4
        elif attn_1_3_output in {9}:
            return position == 2
        elif attn_1_3_output in {18}:
            return position == 9

    num_attn_2_0_pattern = select(positions, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, position):
        if attn_1_1_output in {0, 3, 13, 17, 18}:
            return position == 4
        elif attn_1_1_output in {1, 4}:
            return position == 3
        elif attn_1_1_output in {9, 2, 11, 14}:
            return position == 2
        elif attn_1_1_output in {5, 7}:
            return position == 5
        elif attn_1_1_output in {19, 6}:
            return position == 6
        elif attn_1_1_output in {8}:
            return position == 8
        elif attn_1_1_output in {10}:
            return position == 1
        elif attn_1_1_output in {12, 15}:
            return position == 7
        elif attn_1_1_output in {16}:
            return position == 0

    num_attn_2_1_pattern = select(positions, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_1_output, position):
        if attn_1_1_output in {0, 1, 2, 3, 5, 18}:
            return position == 7
        elif attn_1_1_output in {4}:
            return position == 19
        elif attn_1_1_output in {6}:
            return position == 11
        elif attn_1_1_output in {7}:
            return position == 5
        elif attn_1_1_output in {8, 12}:
            return position == 12
        elif attn_1_1_output in {9}:
            return position == 17
        elif attn_1_1_output in {19, 10, 11}:
            return position == 8
        elif attn_1_1_output in {13}:
            return position == 13
        elif attn_1_1_output in {14}:
            return position == 18
        elif attn_1_1_output in {17, 15}:
            return position == 16
        elif attn_1_1_output in {16}:
            return position == 14

    num_attn_2_2_pattern = select(positions, attn_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_1_1_output, position):
        if num_mlp_1_1_output in {0, 17, 10, 18}:
            return position == 11
        elif num_mlp_1_1_output in {1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 19}:
            return position == 7
        elif num_mlp_1_1_output in {7}:
            return position == 18
        elif num_mlp_1_1_output in {11}:
            return position == 19
        elif num_mlp_1_1_output in {16}:
            return position == 14

    num_attn_2_3_pattern = select(positions, num_mlp_1_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_2_0_output):
        key = (attn_2_1_output, attn_2_0_output)
        return 16

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output):
        key = attn_2_1_output
        return 12

    mlp_2_1_outputs = [mlp_2_1(k0) for k0 in attn_2_1_outputs]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_2_1_output):
        key = (num_attn_1_3_output, num_attn_2_1_output)
        if key in {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)}:
            return 1
        elif key in {(0, 2)}:
            return 10
        return 7

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output):
        key = num_attn_2_2_output
        return 7

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_2_outputs]
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


print(run(["<s>", "0", "3", "3", "3", "1", "3"]))

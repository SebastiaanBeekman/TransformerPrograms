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
        "output/length/rasp/double_hist/trainlength10/s3/double_hist_weights.csv",
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
        if q_position in {0, 5}:
            return k_position == 5
        elif q_position in {1, 3, 4, 7, 8}:
            return k_position == 9
        elif q_position in {2, 6}:
            return k_position == 6
        elif q_position in {16, 9, 18}:
            return k_position == 8
        elif q_position in {10, 15}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 19
        elif q_position in {13}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 15

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 3, 4, 5, 7, 8}:
            return k_position == 7
        elif q_position in {9, 6}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {17, 18, 11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 4
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 6
        elif q_position in {15}:
            return k_position == 12
        elif q_position in {16}:
            return k_position == 16
        elif q_position in {19}:
            return k_position == 19

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "3"
        elif q_token in {"1", "3"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 3, 4, 5, 7, 9}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {10, 12, 13, 14}:
            return k_position == 16
        elif q_position in {17, 11}:
            return k_position == 10
        elif q_position in {19, 15}:
            return k_position == 9
        elif q_position in {16}:
            return k_position == 14
        elif q_position in {18}:
            return k_position == 12

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 11, 12, 13, 16, 17, 18}:
            return k_position == 9
        elif q_position in {1, 2, 3, 4, 5, 7, 8}:
            return k_position == 6
        elif q_position in {10, 19, 6, 15}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 7

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 9, 10, 11, 12, 13, 15, 17, 18}:
            return k_position == 9
        elif q_position in {1, 2, 3, 4, 5, 7, 16}:
            return k_position == 8
        elif q_position in {19, 6, 14}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 5

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
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
            return k_token == "<s>"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
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
            return k_token == "<s>"

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        return 10

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        return 9

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_3_output):
        key = num_attn_0_3_output
        if key in {0, 1}:
            return 15
        elif key in {2}:
            return 10
        return 5

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_3_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_1_output):
        key = (num_attn_0_0_output, num_attn_0_1_output)
        return 3

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, position):
        if attn_0_1_output in {0, 17}:
            return position == 6
        elif attn_0_1_output in {1, 2, 3, 4, 5, 6, 7, 8, 10}:
            return position == 9
        elif attn_0_1_output in {9, 12, 13, 15, 16}:
            return position == 10
        elif attn_0_1_output in {18, 11}:
            return position == 14
        elif attn_0_1_output in {14}:
            return position == 8
        elif attn_0_1_output in {19}:
            return position == 15

    attn_1_0_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"<s>", "2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "1"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, num_mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 8, 11, 13, 14, 16, 17, 18, 19}:
            return position == 6
        elif num_mlp_0_0_output in {1, 3, 9, 7}:
            return position == 10
        elif num_mlp_0_0_output in {2}:
            return position == 19
        elif num_mlp_0_0_output in {4}:
            return position == 14
        elif num_mlp_0_0_output in {5}:
            return position == 0
        elif num_mlp_0_0_output in {6}:
            return position == 12
        elif num_mlp_0_0_output in {10, 15}:
            return position == 4
        elif num_mlp_0_0_output in {12}:
            return position == 5

    attn_1_2_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, positions)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {1, 3}:
            return k_num_mlp_0_0_output == 6
        elif q_num_mlp_0_0_output in {2, 5}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {4, 6}:
            return k_num_mlp_0_0_output == 7
        elif q_num_mlp_0_0_output in {7}:
            return k_num_mlp_0_0_output == 17
        elif q_num_mlp_0_0_output in {8, 9, 10}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {17, 11}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {12}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {16, 13}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {14, 15}:
            return k_num_mlp_0_0_output == 5
        elif q_num_mlp_0_0_output in {18}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {19}:
            return k_num_mlp_0_0_output == 12

    attn_1_3_pattern = select_closest(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, predicate_1_3
    )
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 1, 12}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {2, 11, 16, 18, 19}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {3, 13, 7}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {10, 4, 6}:
            return k_num_mlp_0_0_output == 18
        elif q_num_mlp_0_0_output in {5}:
            return k_num_mlp_0_0_output == 6
        elif q_num_mlp_0_0_output in {8}:
            return k_num_mlp_0_0_output == 11
        elif q_num_mlp_0_0_output in {9}:
            return k_num_mlp_0_0_output == 19
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {15}:
            return k_num_mlp_0_0_output == 15
        elif q_num_mlp_0_0_output in {17}:
            return k_num_mlp_0_0_output == 17

    num_attn_1_0_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 1, 6, 16}:
            return position == 8
        elif num_mlp_0_0_output in {2, 3, 4, 7, 10, 11, 17}:
            return position == 9
        elif num_mlp_0_0_output in {5}:
            return position == 1
        elif num_mlp_0_0_output in {8}:
            return position == 15
        elif num_mlp_0_0_output in {9}:
            return position == 13
        elif num_mlp_0_0_output in {18, 12, 15}:
            return position == 7
        elif num_mlp_0_0_output in {13}:
            return position == 16
        elif num_mlp_0_0_output in {14}:
            return position == 14
        elif num_mlp_0_0_output in {19}:
            return position == 5

    num_attn_1_1_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, attn_0_0_output):
        if num_mlp_0_0_output in {0, 5}:
            return attn_0_0_output == 13
        elif num_mlp_0_0_output in {1}:
            return attn_0_0_output == 17
        elif num_mlp_0_0_output in {8, 16, 2, 10}:
            return attn_0_0_output == 0
        elif num_mlp_0_0_output in {3}:
            return attn_0_0_output == 12
        elif num_mlp_0_0_output in {4, 12}:
            return attn_0_0_output == 16
        elif num_mlp_0_0_output in {11, 6, 7}:
            return attn_0_0_output == 8
        elif num_mlp_0_0_output in {9}:
            return attn_0_0_output == 18
        elif num_mlp_0_0_output in {18, 13}:
            return attn_0_0_output == 19
        elif num_mlp_0_0_output in {19, 14}:
            return attn_0_0_output == 14
        elif num_mlp_0_0_output in {15}:
            return attn_0_0_output == 9
        elif num_mlp_0_0_output in {17}:
            return attn_0_0_output == 11

    num_attn_1_2_pattern = select(
        attn_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "0"
        elif attn_0_2_output in {"1"}:
            return token == "1"
        elif attn_0_2_output in {"2"}:
            return token == "2"
        elif attn_0_2_output in {"<s>", "3"}:
            return token == "3"
        elif attn_0_2_output in {"4"}:
            return token == "4"
        elif attn_0_2_output in {"5"}:
            return token == "5"

    num_attn_1_3_pattern = select(tokens, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_1_output):
        key = (attn_1_1_output, attn_0_1_output)
        if key in {
            (0, 0),
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 7),
            (0, 9),
            (0, 11),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 18),
            (3, 14),
            (4, 14),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 7),
            (5, 9),
            (5, 11),
            (5, 13),
            (5, 14),
            (5, 15),
            (5, 18),
            (6, 14),
            (7, 14),
            (8, 14),
            (9, 14),
            (12, 14),
            (17, 14),
            (18, 14),
            (19, 14),
        }:
            return 18
        return 8

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, attn_1_3_output):
        key = (attn_1_0_output, attn_1_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
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
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
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
            (2, 10),
            (3, 1),
            (3, 3),
            (3, 4),
            (3, 10),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 10),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (11, 1),
            (11, 3),
            (11, 10),
            (12, 1),
            (12, 3),
            (12, 10),
            (13, 1),
            (13, 3),
            (13, 10),
            (14, 1),
            (14, 3),
            (14, 4),
            (14, 10),
            (15, 1),
            (15, 10),
            (16, 1),
            (16, 3),
            (16, 10),
            (17, 1),
            (17, 3),
            (17, 10),
            (18, 1),
            (18, 3),
            (18, 10),
            (19, 1),
            (19, 3),
            (19, 4),
            (19, 10),
        }:
            return 5
        elif key in {
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
            (5, 9),
            (5, 12),
            (5, 15),
            (5, 18),
            (5, 19),
            (7, 3),
            (7, 6),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
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
            (9, 3),
            (9, 6),
            (10, 6),
            (11, 6),
            (12, 6),
            (13, 6),
            (14, 6),
            (15, 3),
            (15, 6),
            (16, 6),
            (17, 6),
            (18, 6),
            (19, 6),
        }:
            return 8
        elif key in {
            (1, 0),
            (2, 0),
            (3, 0),
            (3, 2),
            (3, 8),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (3, 19),
            (4, 0),
            (5, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
        }:
            return 6
        elif key in {
            (0, 5),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 10),
            (5, 13),
            (10, 5),
            (11, 5),
            (12, 5),
            (13, 5),
            (14, 5),
            (16, 5),
            (17, 5),
            (18, 5),
            (19, 5),
        }:
            return 4
        elif key in {
            (9, 10),
            (11, 4),
            (12, 4),
            (13, 4),
            (14, 2),
            (16, 4),
            (17, 4),
            (18, 4),
        }:
            return 16
        return 15

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_3_output, num_attn_1_1_output):
        key = (num_attn_0_3_output, num_attn_1_1_output)
        return 11

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output, num_attn_1_3_output):
        key = (num_attn_0_2_output, num_attn_1_3_output)
        if key in {(0, 0)}:
            return 18
        return 9

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_2_output, num_mlp_0_0_output):
        if attn_1_2_output in {0, 1, 2, 3, 4, 6, 7, 8}:
            return num_mlp_0_0_output == 5
        elif attn_1_2_output in {5}:
            return num_mlp_0_0_output == 3
        elif attn_1_2_output in {9}:
            return num_mlp_0_0_output == 2
        elif attn_1_2_output in {10}:
            return num_mlp_0_0_output == 9
        elif attn_1_2_output in {11, 13, 17, 18, 19}:
            return num_mlp_0_0_output == 7
        elif attn_1_2_output in {12, 14, 15}:
            return num_mlp_0_0_output == 6
        elif attn_1_2_output in {16}:
            return num_mlp_0_0_output == 11

    attn_2_0_pattern = select_closest(
        num_mlp_0_0_outputs, attn_1_2_outputs, predicate_2_0
    )
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_1_output, position):
        if attn_1_1_output in {0, 8, 2, 14}:
            return position == 6
        elif attn_1_1_output in {1, 13}:
            return position == 9
        elif attn_1_1_output in {3, 4, 9, 10, 12, 15, 17, 18, 19}:
            return position == 8
        elif attn_1_1_output in {5}:
            return position == 5
        elif attn_1_1_output in {6}:
            return position == 11
        elif attn_1_1_output in {7}:
            return position == 16
        elif attn_1_1_output in {11}:
            return position == 13
        elif attn_1_1_output in {16}:
            return position == 7

    attn_2_1_pattern = select_closest(positions, attn_1_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, num_mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 10, 11, 12, 14, 17, 18, 19}:
            return k_position == 10
        elif q_position in {1, 2, 3, 4, 5, 7, 13, 15, 16}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 9
        elif q_position in {8, 9}:
            return k_position == 2

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, num_mlp_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 1, 3, 8, 11, 12, 14, 15, 16, 17, 19}:
            return k_num_mlp_0_0_output == 10
        elif q_num_mlp_0_0_output in {2, 4, 7, 9, 13, 18}:
            return k_num_mlp_0_0_output == 5
        elif q_num_mlp_0_0_output in {10, 5}:
            return k_num_mlp_0_0_output == 15
        elif q_num_mlp_0_0_output in {6}:
            return k_num_mlp_0_0_output == 2

    num_attn_2_0_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_0
    )
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, num_mlp_0_0_output):
        if attn_1_1_output in {0, 1, 2, 3, 4, 6, 9, 10, 11, 17}:
            return num_mlp_0_0_output == 10
        elif attn_1_1_output in {8, 5}:
            return num_mlp_0_0_output == 15
        elif attn_1_1_output in {7}:
            return num_mlp_0_0_output == 19
        elif attn_1_1_output in {12, 13, 14, 15, 16}:
            return num_mlp_0_0_output == 9
        elif attn_1_1_output in {18, 19}:
            return num_mlp_0_0_output == 13

    num_attn_2_1_pattern = select(
        num_mlp_0_0_outputs, attn_1_1_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_0_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, attn_1_2_output):
        if attn_1_0_output in {0, 5, 6, 7}:
            return attn_1_2_output == 0
        elif attn_1_0_output in {1, 2, 4}:
            return attn_1_2_output == 2
        elif attn_1_0_output in {3}:
            return attn_1_2_output == 1
        elif attn_1_0_output in {8}:
            return attn_1_2_output == 19
        elif attn_1_0_output in {9, 13}:
            return attn_1_2_output == 17
        elif attn_1_0_output in {10, 11, 12, 14, 16, 17, 18, 19}:
            return attn_1_2_output == 10
        elif attn_1_0_output in {15}:
            return attn_1_2_output == 18

    num_attn_2_2_pattern = select(attn_1_2_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_0_0_output, attn_1_1_output):
        if num_mlp_0_0_output in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 18, 19}:
            return attn_1_1_output == 10
        elif num_mlp_0_0_output in {10, 15}:
            return attn_1_1_output == 5
        elif num_mlp_0_0_output in {11}:
            return attn_1_1_output == 1
        elif num_mlp_0_0_output in {17}:
            return attn_1_1_output == 0

    num_attn_2_3_pattern = select(
        attn_1_1_outputs, num_mlp_0_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_2_output, attn_2_0_output):
        key = (attn_0_2_output, attn_2_0_output)
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        return 8

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 19

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
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


print(run(["<s>", "1", "3", "0", "0", "0", "5", "5", "3", "2"]))

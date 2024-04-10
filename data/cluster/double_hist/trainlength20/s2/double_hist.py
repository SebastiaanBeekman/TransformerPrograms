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
        "output/length/rasp/double_hist/trainlength20/s2/double_hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"<s>", "4", "1"}:
            return k_token == "0"
        elif q_token in {"2", "5"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "5"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 18}:
            return k_position == 12
        elif q_position in {1, 2, 3, 4}:
            return k_position == 5
        elif q_position in {5, 6, 21, 23, 24, 25}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {10, 27, 12}:
            return k_position == 9
        elif q_position in {11, 29, 22}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 10
        elif q_position in {15, 16, 17, 19, 28}:
            return k_position == 14
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {26}:
            return k_position == 24

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 18}:
            return k_position == 12
        elif q_position in {1, 7}:
            return k_position == 7
        elif q_position in {27, 2, 3, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 1
        elif q_position in {8, 23}:
            return k_position == 8
        elif q_position in {9}:
            return k_position == 2
        elif q_position in {25, 10, 12}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {13, 14}:
            return k_position == 13
        elif q_position in {16, 17, 15}:
            return k_position == 14
        elif q_position in {19}:
            return k_position == 10
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {24, 28, 21, 22}:
            return k_position == 5
        elif q_position in {26}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 18

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 16, 17}:
            return k_position == 12
        elif q_position in {1, 6}:
            return k_position == 4
        elif q_position in {2, 3, 4, 5, 7, 8, 9}:
            return k_position == 6
        elif q_position in {10, 18, 13, 14}:
            return k_position == 14
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {25, 15}:
            return k_position == 13
        elif q_position in {19}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 26
        elif q_position in {21}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {26, 23}:
            return k_position == 11
        elif q_position in {24}:
            return k_position == 16
        elif q_position in {27}:
            return k_position == 7
        elif q_position in {28}:
            return k_position == 17
        elif q_position in {29}:
            return k_position == 25

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
            return k_position == 1
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {9, 3, 4}:
            return k_position == 2
        elif q_position in {10, 5, 6}:
            return k_position == 0
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 5
        elif q_position in {13, 14}:
            return k_position == 13
        elif q_position in {16, 27, 22, 15}:
            return k_position == 28
        elif q_position in {17, 20}:
            return k_position == 26
        elif q_position in {18}:
            return k_position == 29
        elif q_position in {19}:
            return k_position == 21
        elif q_position in {28, 21}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 18
        elif q_position in {25, 29}:
            return k_position == 25
        elif q_position in {26}:
            return k_position == 17

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"5", "0", "3", "4", "1", "2"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
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
    def mlp_0_0(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        if key in {
            (0, 0),
            (0, 3),
            (0, 4),
            (0, 19),
            (2, 0),
            (2, 3),
            (2, 4),
            (2, 19),
            (3, 19),
            (20, 19),
            (21, 19),
            (22, 19),
            (23, 19),
            (24, 19),
            (25, 19),
            (26, 19),
            (27, 19),
            (28, 19),
            (29, 19),
        }:
            return 2
        return 17

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_0_output):
        key = (token, attn_0_0_output)
        return 13

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_0_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        if key in {
            (0, 0),
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
            (1, 0),
            (1, 1),
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
        }:
            return 16
        elif key in {
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (2, 0),
            (2, 1),
            (2, 2),
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
        }:
            return 9
        return 23

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 13

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"5", "<s>", "4"}:
            return k_token == "2"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0}:
            return position == 9
        elif num_mlp_0_0_output in {1, 9}:
            return position == 13
        elif num_mlp_0_0_output in {2}:
            return position == 12
        elif num_mlp_0_0_output in {3, 7, 8, 12, 23}:
            return position == 20
        elif num_mlp_0_0_output in {4, 5, 17, 18, 22, 25}:
            return position == 8
        elif num_mlp_0_0_output in {6}:
            return position == 24
        elif num_mlp_0_0_output in {10}:
            return position == 17
        elif num_mlp_0_0_output in {11}:
            return position == 22
        elif num_mlp_0_0_output in {13}:
            return position == 14
        elif num_mlp_0_0_output in {14}:
            return position == 10
        elif num_mlp_0_0_output in {15}:
            return position == 27
        elif num_mlp_0_0_output in {16, 20}:
            return position == 11
        elif num_mlp_0_0_output in {19, 21, 24, 26, 28, 29}:
            return position == 7
        elif num_mlp_0_0_output in {27}:
            return position == 19

    attn_1_1_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {0}:
            return position == 4
        elif attn_0_1_output in {1, 4, 9, 11, 15, 16}:
            return position == 10
        elif attn_0_1_output in {8, 2, 3, 5}:
            return position == 9
        elif attn_0_1_output in {6}:
            return position == 11
        elif attn_0_1_output in {10, 19, 7}:
            return position == 14
        elif attn_0_1_output in {12}:
            return position == 13
        elif attn_0_1_output in {13}:
            return position == 15
        elif attn_0_1_output in {18, 14}:
            return position == 12
        elif attn_0_1_output in {24, 17}:
            return position == 28
        elif attn_0_1_output in {20, 28}:
            return position == 6
        elif attn_0_1_output in {21}:
            return position == 5
        elif attn_0_1_output in {29, 22}:
            return position == 26
        elif attn_0_1_output in {23}:
            return position == 0
        elif attn_0_1_output in {25}:
            return position == 23
        elif attn_0_1_output in {26}:
            return position == 16
        elif attn_0_1_output in {27}:
            return position == 24

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 16}:
            return position == 5
        elif num_mlp_0_0_output in {1, 2, 5}:
            return position == 9
        elif num_mlp_0_0_output in {28, 3, 4, 21}:
            return position == 11
        elif num_mlp_0_0_output in {6, 9, 15, 19, 23}:
            return position == 10
        elif num_mlp_0_0_output in {7}:
            return position == 13
        elif num_mlp_0_0_output in {8}:
            return position == 25
        elif num_mlp_0_0_output in {24, 10, 20, 29}:
            return position == 8
        elif num_mlp_0_0_output in {11, 12}:
            return position == 14
        elif num_mlp_0_0_output in {13}:
            return position == 15
        elif num_mlp_0_0_output in {14}:
            return position == 12
        elif num_mlp_0_0_output in {17}:
            return position == 7
        elif num_mlp_0_0_output in {27, 18, 26}:
            return position == 6
        elif num_mlp_0_0_output in {22}:
            return position == 17
        elif num_mlp_0_0_output in {25}:
            return position == 16

    attn_1_3_pattern = select_closest(positions, num_mlp_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 24}:
            return position == 11
        elif num_mlp_0_0_output in {1, 4, 10, 17, 18}:
            return position == 29
        elif num_mlp_0_0_output in {2, 12, 14, 20, 21, 28}:
            return position == 28
        elif num_mlp_0_0_output in {3, 15}:
            return position == 24
        elif num_mlp_0_0_output in {8, 25, 5, 22}:
            return position == 21
        elif num_mlp_0_0_output in {16, 6}:
            return position == 27
        elif num_mlp_0_0_output in {7}:
            return position == 20
        elif num_mlp_0_0_output in {9}:
            return position == 7
        elif num_mlp_0_0_output in {11, 29}:
            return position == 22
        elif num_mlp_0_0_output in {26, 13}:
            return position == 6
        elif num_mlp_0_0_output in {27, 19}:
            return position == 25
        elif num_mlp_0_0_output in {23}:
            return position == 9

    num_attn_1_0_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 9, 26}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {1, 28}:
            return k_num_mlp_0_0_output == 20
        elif q_num_mlp_0_0_output in {2, 7}:
            return k_num_mlp_0_0_output == 17
        elif q_num_mlp_0_0_output in {3}:
            return k_num_mlp_0_0_output == 14
        elif q_num_mlp_0_0_output in {4}:
            return k_num_mlp_0_0_output == 23
        elif q_num_mlp_0_0_output in {5}:
            return k_num_mlp_0_0_output == 29
        elif q_num_mlp_0_0_output in {8, 6}:
            return k_num_mlp_0_0_output == 24
        elif q_num_mlp_0_0_output in {10, 15}:
            return k_num_mlp_0_0_output == 8
        elif q_num_mlp_0_0_output in {11, 16, 17, 22, 27}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {18, 12, 29}:
            return k_num_mlp_0_0_output == 12
        elif q_num_mlp_0_0_output in {13}:
            return k_num_mlp_0_0_output == 18
        elif q_num_mlp_0_0_output in {14}:
            return k_num_mlp_0_0_output == 21
        elif q_num_mlp_0_0_output in {25, 19}:
            return k_num_mlp_0_0_output == 22
        elif q_num_mlp_0_0_output in {20}:
            return k_num_mlp_0_0_output == 27
        elif q_num_mlp_0_0_output in {21}:
            return k_num_mlp_0_0_output == 0
        elif q_num_mlp_0_0_output in {23}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {24}:
            return k_num_mlp_0_0_output == 25

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 2, 13}:
            return position == 11
        elif num_mlp_0_0_output in {16, 1, 6}:
            return position == 21
        elif num_mlp_0_0_output in {3, 20, 28}:
            return position == 26
        elif num_mlp_0_0_output in {4, 5}:
            return position == 7
        elif num_mlp_0_0_output in {27, 21, 7}:
            return position == 20
        elif num_mlp_0_0_output in {8, 17, 11}:
            return position == 27
        elif num_mlp_0_0_output in {9, 26}:
            return position == 6
        elif num_mlp_0_0_output in {25, 10, 22}:
            return position == 29
        elif num_mlp_0_0_output in {12}:
            return position == 24
        elif num_mlp_0_0_output in {14}:
            return position == 22
        elif num_mlp_0_0_output in {18, 15}:
            return position == 28
        elif num_mlp_0_0_output in {19, 23}:
            return position == 15
        elif num_mlp_0_0_output in {24}:
            return position == 13
        elif num_mlp_0_0_output in {29}:
            return position == 16

    num_attn_1_2_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_3_output, position):
        if attn_0_3_output in {0, 2, 3}:
            return position == 3
        elif attn_0_3_output in {1, 8, 20, 22, 23, 25, 26, 27}:
            return position == 1
        elif attn_0_3_output in {10, 4}:
            return position == 4
        elif attn_0_3_output in {5}:
            return position == 5
        elif attn_0_3_output in {6}:
            return position == 6
        elif attn_0_3_output in {7}:
            return position == 7
        elif attn_0_3_output in {9}:
            return position == 9
        elif attn_0_3_output in {11}:
            return position == 11
        elif attn_0_3_output in {12, 13, 14, 15}:
            return position == 21
        elif attn_0_3_output in {16, 19}:
            return position == 27
        elif attn_0_3_output in {17}:
            return position == 22
        elif attn_0_3_output in {18, 29}:
            return position == 28
        elif attn_0_3_output in {21}:
            return position == 0
        elif attn_0_3_output in {24}:
            return position == 2
        elif attn_0_3_output in {28}:
            return position == 25

    num_attn_1_3_pattern = select(positions, attn_0_3_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, attn_0_2_output):
        key = (attn_1_1_output, attn_0_2_output)
        return 28

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        if key in {
            (0, 6),
            (0, 10),
            (0, 16),
            (1, 16),
            (2, 0),
            (2, 2),
            (2, 4),
            (2, 5),
            (2, 6),
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
            (3, 16),
            (4, 16),
            (5, 16),
            (6, 16),
            (7, 0),
            (7, 6),
            (7, 10),
            (7, 11),
            (7, 13),
            (7, 14),
            (7, 16),
            (7, 19),
            (7, 21),
            (7, 27),
            (9, 16),
            (10, 0),
            (10, 2),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 8),
            (10, 9),
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
            (11, 16),
            (12, 16),
            (13, 16),
            (16, 0),
            (16, 2),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 8),
            (16, 9),
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
            (17, 16),
            (18, 0),
            (18, 6),
            (18, 10),
            (18, 13),
            (18, 14),
            (18, 16),
            (18, 19),
            (18, 21),
            (19, 16),
            (20, 0),
            (20, 6),
            (20, 10),
            (20, 13),
            (20, 16),
            (20, 19),
            (20, 21),
            (21, 16),
            (22, 16),
            (23, 16),
            (24, 6),
            (24, 10),
            (24, 13),
            (24, 16),
            (24, 19),
            (24, 21),
            (25, 16),
            (26, 0),
            (26, 6),
            (26, 10),
            (26, 13),
            (26, 14),
            (26, 16),
            (26, 19),
            (26, 21),
            (27, 6),
            (27, 10),
            (27, 13),
            (27, 16),
            (27, 19),
            (27, 21),
            (28, 6),
            (28, 10),
            (28, 16),
            (29, 16),
        }:
            return 14
        return 29

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_1_output):
        key = (num_attn_1_2_output, num_attn_1_1_output)
        return 11

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 24

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "5", "0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"3", "4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_3_output, num_mlp_0_0_output):
        if attn_0_3_output in {0}:
            return num_mlp_0_0_output == 8
        elif attn_0_3_output in {8, 1, 3, 5}:
            return num_mlp_0_0_output == 11
        elif attn_0_3_output in {2, 18}:
            return num_mlp_0_0_output == 19
        elif attn_0_3_output in {4, 13, 6, 14}:
            return num_mlp_0_0_output == 23
        elif attn_0_3_output in {26, 22, 7}:
            return num_mlp_0_0_output == 12
        elif attn_0_3_output in {9}:
            return num_mlp_0_0_output == 16
        elif attn_0_3_output in {10, 11, 12, 15, 17, 19, 21}:
            return num_mlp_0_0_output == 13
        elif attn_0_3_output in {16}:
            return num_mlp_0_0_output == 26
        elif attn_0_3_output in {24, 25, 20}:
            return num_mlp_0_0_output == 6
        elif attn_0_3_output in {23}:
            return num_mlp_0_0_output == 1
        elif attn_0_3_output in {27}:
            return num_mlp_0_0_output == 21
        elif attn_0_3_output in {28}:
            return num_mlp_0_0_output == 18
        elif attn_0_3_output in {29}:
            return num_mlp_0_0_output == 5

    attn_2_1_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_3_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 19, 24, 25, 26, 29}:
            return k_position == 6
        elif q_position in {1, 5, 6, 7}:
            return k_position == 9
        elif q_position in {2, 4, 10, 11, 13, 17}:
            return k_position == 7
        elif q_position in {9, 3, 23}:
            return k_position == 1
        elif q_position in {8, 12, 14, 18, 21, 27}:
            return k_position == 5
        elif q_position in {15}:
            return k_position == 8
        elif q_position in {16, 20}:
            return k_position == 4
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 14

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, num_mlp_0_0_output):
        if attn_0_1_output in {0, 7, 10, 22, 25, 27, 29}:
            return num_mlp_0_0_output == 1
        elif attn_0_1_output in {1, 2, 3, 5, 6, 8, 11, 21, 24, 28}:
            return num_mlp_0_0_output == 23
        elif attn_0_1_output in {26, 4, 20}:
            return num_mlp_0_0_output == 13
        elif attn_0_1_output in {9}:
            return num_mlp_0_0_output == 3
        elif attn_0_1_output in {12, 14, 15, 16, 17, 18, 19}:
            return num_mlp_0_0_output == 26
        elif attn_0_1_output in {13}:
            return num_mlp_0_0_output == 29
        elif attn_0_1_output in {23}:
            return num_mlp_0_0_output == 9

    attn_2_3_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_1_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_2_output):
        if attn_1_2_output in {0, 1, 2, 21, 27}:
            return attn_0_2_output == 0
        elif attn_1_2_output in {3, 22, 23}:
            return attn_0_2_output == 3
        elif attn_1_2_output in {4}:
            return attn_0_2_output == 4
        elif attn_1_2_output in {5}:
            return attn_0_2_output == 5
        elif attn_1_2_output in {6}:
            return attn_0_2_output == 6
        elif attn_1_2_output in {29, 7}:
            return attn_0_2_output == 28
        elif attn_1_2_output in {8, 19}:
            return attn_0_2_output == 29
        elif attn_1_2_output in {9, 12}:
            return attn_0_2_output == 20
        elif attn_1_2_output in {10, 18, 20, 25, 28}:
            return attn_0_2_output == 27
        elif attn_1_2_output in {11, 13}:
            return attn_0_2_output == 24
        elif attn_1_2_output in {14}:
            return attn_0_2_output == 25
        elif attn_1_2_output in {15}:
            return attn_0_2_output == 23
        elif attn_1_2_output in {16, 17}:
            return attn_0_2_output == 26
        elif attn_1_2_output in {24, 26}:
            return attn_0_2_output == 22

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 24, 3}:
            return k_num_mlp_0_0_output == 21
        elif q_num_mlp_0_0_output in {8, 1, 19, 21}:
            return k_num_mlp_0_0_output == 27
        elif q_num_mlp_0_0_output in {2, 9, 10, 13, 16, 17, 25, 26, 27}:
            return k_num_mlp_0_0_output == 23
        elif q_num_mlp_0_0_output in {4, 6, 14}:
            return k_num_mlp_0_0_output == 20
        elif q_num_mlp_0_0_output in {5}:
            return k_num_mlp_0_0_output == 24
        elif q_num_mlp_0_0_output in {20, 7}:
            return k_num_mlp_0_0_output == 22
        elif q_num_mlp_0_0_output in {11, 15}:
            return k_num_mlp_0_0_output == 28
        elif q_num_mlp_0_0_output in {18, 12}:
            return k_num_mlp_0_0_output == 29
        elif q_num_mlp_0_0_output in {22}:
            return k_num_mlp_0_0_output == 25
        elif q_num_mlp_0_0_output in {23}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {28}:
            return k_num_mlp_0_0_output == 15
        elif q_num_mlp_0_0_output in {29}:
            return k_num_mlp_0_0_output == 19

    num_attn_2_1_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0}:
            return position == 10
        elif num_mlp_0_0_output in {1, 3, 29}:
            return position == 26
        elif num_mlp_0_0_output in {19, 2, 11, 20}:
            return position == 20
        elif num_mlp_0_0_output in {27, 4, 7}:
            return position == 22
        elif num_mlp_0_0_output in {5}:
            return position == 8
        elif num_mlp_0_0_output in {12, 21, 6}:
            return position == 21
        elif num_mlp_0_0_output in {8, 9, 13}:
            return position == 7
        elif num_mlp_0_0_output in {10, 22}:
            return position == 24
        elif num_mlp_0_0_output in {14}:
            return position == 29
        elif num_mlp_0_0_output in {16, 15}:
            return position == 28
        elif num_mlp_0_0_output in {17, 28, 25}:
            return position == 27
        elif num_mlp_0_0_output in {18}:
            return position == 25
        elif num_mlp_0_0_output in {23}:
            return position == 13
        elif num_mlp_0_0_output in {24, 26}:
            return position == 6

    num_attn_2_2_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
            11,
            12,
            14,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            27,
            28,
            29,
        }:
            return k_num_mlp_0_0_output == 23
        elif q_num_mlp_0_0_output in {1}:
            return k_num_mlp_0_0_output == 0
        elif q_num_mlp_0_0_output in {9, 13, 26, 5}:
            return k_num_mlp_0_0_output == 16
        elif q_num_mlp_0_0_output in {17, 15}:
            return k_num_mlp_0_0_output == 9
        elif q_num_mlp_0_0_output in {16}:
            return k_num_mlp_0_0_output == 13
        elif q_num_mlp_0_0_output in {23}:
            return k_num_mlp_0_0_output == 26

    num_attn_2_3_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_3
    )
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_2_output, attn_2_0_output):
        key = (attn_2_2_output, attn_2_0_output)
        return 23

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_2_outputs, attn_2_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_0_1_output, attn_1_1_output):
        key = (mlp_0_1_output, attn_1_1_output)
        if key in {
            (2, 20),
            (3, 20),
            (5, 20),
            (7, 20),
            (8, 5),
            (8, 8),
            (8, 11),
            (8, 18),
            (8, 20),
            (9, 20),
            (14, 20),
            (16, 5),
            (16, 8),
            (16, 11),
            (16, 20),
            (18, 20),
            (19, 20),
            (20, 20),
            (21, 20),
            (22, 5),
            (22, 8),
            (22, 11),
            (22, 18),
            (22, 20),
            (23, 20),
            (24, 20),
            (25, 5),
            (25, 8),
            (25, 11),
            (25, 18),
            (25, 20),
            (26, 20),
            (27, 20),
            (29, 20),
        }:
            return 5
        return 3

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_0_1_outputs, attn_1_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_2_0_output):
        key = (num_attn_1_0_output, num_attn_2_0_output)
        return 28

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 27

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_0_outputs]
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


print(run(["<s>", "5", "0", "3", "2", "3", "0", "2", "1", "3"]))

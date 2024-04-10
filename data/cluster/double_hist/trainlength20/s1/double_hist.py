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
        "output/length/rasp/double_hist/trainlength20/s1/double_hist_weights.csv",
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
            return k_position == 14
        elif q_position in {1, 2, 3, 16, 29}:
            return k_position == 5
        elif q_position in {18, 4, 23}:
            return k_position == 10
        elif q_position in {17, 5}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {10, 7}:
            return k_position == 4
        elif q_position in {8, 24}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {12, 14}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 17
        elif q_position in {19, 21}:
            return k_position == 8
        elif q_position in {25, 20}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 15
        elif q_position in {27}:
            return k_position == 25
        elif q_position in {28}:
            return k_position == 29

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"5", "1"}:
            return k_token == "2"
        elif q_token in {"3", "2"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 1}:
            return k_position == 4
        elif q_position in {2, 3, 4, 6, 10}:
            return k_position == 7
        elif q_position in {21, 5}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 10
        elif q_position in {8, 9}:
            return k_position == 9
        elif q_position in {17, 11}:
            return k_position == 12
        elif q_position in {27, 19, 12}:
            return k_position == 13
        elif q_position in {13, 20, 22, 25, 26, 28, 29}:
            return k_position == 5
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {24, 15}:
            return k_position == 11
        elif q_position in {16, 18}:
            return k_position == 8
        elif q_position in {23}:
            return k_position == 25

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "5"
        elif q_token in {"<s>", "3"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "4"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0", "4", "2", "5", "1", "<s>", "3"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 12
        elif token in {"2", "1"}:
            return position == 14
        elif token in {"3"}:
            return position == 17
        elif token in {"4", "5"}:
            return position == 15
        elif token in {"<s>"}:
            return position == 27

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0", "4", "2", "5", "1", "<s>", "3"}:
            return k_token == ""

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
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        if key in {
            (12, 1),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (13, 6),
            (13, 7),
            (13, 8),
            (13, 9),
            (13, 10),
            (13, 11),
            (13, 12),
            (13, 13),
            (13, 14),
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
            (15, 1),
            (16, 1),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 9),
            (17, 10),
            (17, 11),
            (17, 12),
            (17, 13),
            (17, 14),
            (17, 15),
            (17, 16),
            (17, 17),
            (17, 18),
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
            (19, 1),
        }:
            return 16
        return 8

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, token):
        key = (attn_0_2_output, token)
        return 7

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {
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
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
        }:
            return 27
        return 16

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        return 17

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_2_output, position):
        if attn_0_2_output in {0, 3, 12, 14}:
            return position == 8
        elif attn_0_2_output in {1, 4, 6}:
            return position == 7
        elif attn_0_2_output in {17, 2}:
            return position == 12
        elif attn_0_2_output in {9, 5, 7}:
            return position == 6
        elif attn_0_2_output in {8, 18, 19}:
            return position == 11
        elif attn_0_2_output in {10, 13}:
            return position == 9
        elif attn_0_2_output in {24, 11, 21}:
            return position == 5
        elif attn_0_2_output in {15}:
            return position == 13
        elif attn_0_2_output in {16}:
            return position == 17
        elif attn_0_2_output in {20}:
            return position == 27
        elif attn_0_2_output in {22}:
            return position == 4
        elif attn_0_2_output in {25, 27, 23}:
            return position == 20
        elif attn_0_2_output in {26}:
            return position == 24
        elif attn_0_2_output in {28}:
            return position == 21
        elif attn_0_2_output in {29}:
            return position == 23

    attn_1_0_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"2", "1"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4", "5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {8, 6}:
            return k_position == 2
        elif q_position in {16, 24, 7}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 1
        elif q_position in {10, 18, 21, 22, 25, 26, 28}:
            return k_position == 9
        elif q_position in {11, 14}:
            return k_position == 10
        elif q_position in {12, 15}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {19, 23}:
            return k_position == 13
        elif q_position in {20}:
            return k_position == 26
        elif q_position in {27}:
            return k_position == 12
        elif q_position in {29}:
            return k_position == 20

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, positions)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"0", "3", "1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "5"
        elif q_token in {"4", "<s>"}:
            return k_token == ""
        elif q_token in {"5"}:
            return k_token == "0"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {
            0,
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            14,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif attn_0_3_output in {3}:
            return token == "2"
        elif attn_0_3_output in {16, 15, 13, 7}:
            return token == "<pad>"

    num_attn_1_0_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, position):
        if attn_0_3_output in {0, 1, 2, 3, 22, 26}:
            return position == 13
        elif attn_0_3_output in {4, 5, 6, 9, 12, 24, 27}:
            return position == 19
        elif attn_0_3_output in {14, 7}:
            return position == 23
        elif attn_0_3_output in {8}:
            return position == 25
        elif attn_0_3_output in {10, 28}:
            return position == 20
        elif attn_0_3_output in {29, 11, 21, 23}:
            return position == 15
        elif attn_0_3_output in {13}:
            return position == 22
        elif attn_0_3_output in {15}:
            return position == 26
        elif attn_0_3_output in {16}:
            return position == 27
        elif attn_0_3_output in {17}:
            return position == 21
        elif attn_0_3_output in {18}:
            return position == 28
        elif attn_0_3_output in {19}:
            return position == 24
        elif attn_0_3_output in {20}:
            return position == 0
        elif attn_0_3_output in {25}:
            return position == 16

    num_attn_1_1_pattern = select(positions, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_2_output, position):
        if attn_0_2_output in {0, 24, 29}:
            return position == 21
        elif attn_0_2_output in {1, 2, 3, 13}:
            return position == 28
        elif attn_0_2_output in {16, 17, 18, 4}:
            return position == 24
        elif attn_0_2_output in {19, 20, 5, 7}:
            return position == 26
        elif attn_0_2_output in {12, 23, 6, 22}:
            return position == 29
        elif attn_0_2_output in {8, 26, 28, 21}:
            return position == 22
        elif attn_0_2_output in {9, 14}:
            return position == 20
        elif attn_0_2_output in {10}:
            return position == 23
        elif attn_0_2_output in {27, 25, 11}:
            return position == 27
        elif attn_0_2_output in {15}:
            return position == 19

    num_attn_1_2_pattern = select(positions, attn_0_2_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 9
        elif attn_0_2_output in {1, 13}:
            return position == 17
        elif attn_0_2_output in {2}:
            return position == 7
        elif attn_0_2_output in {25, 3}:
            return position == 27
        elif attn_0_2_output in {4}:
            return position == 26
        elif attn_0_2_output in {21, 19, 20, 5}:
            return position == 23
        elif attn_0_2_output in {8, 6}:
            return position == 25
        elif attn_0_2_output in {7}:
            return position == 20
        elif attn_0_2_output in {9, 23}:
            return position == 28
        elif attn_0_2_output in {16, 10, 18}:
            return position == 29
        elif attn_0_2_output in {11, 14}:
            return position == 22
        elif attn_0_2_output in {12, 29}:
            return position == 24
        elif attn_0_2_output in {15, 17, 22, 26, 28}:
            return position == 21
        elif attn_0_2_output in {24}:
            return position == 0
        elif attn_0_2_output in {27}:
            return position == 19

    num_attn_1_3_pattern = select(positions, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_1_output, attn_1_3_output):
        key = (num_mlp_0_1_output, attn_1_3_output)
        if key in {
            (15, 1),
            (15, 4),
            (15, 9),
            (15, 10),
            (15, 21),
            (15, 23),
            (15, 28),
            (25, 9),
            (25, 10),
            (25, 21),
            (25, 23),
            (25, 28),
        }:
            return 3
        return 19

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_0_output):
        key = (attn_1_2_output, attn_1_0_output)
        return 25

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        if key in {
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
        }:
            return 9
        return 10

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_0_1_output):
        key = (num_attn_1_1_output, num_attn_0_1_output)
        return 26

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 15, 25, 29}:
            return k_position == 5
        elif q_position in {5, 7, 9, 12, 19}:
            return k_position == 8
        elif q_position in {16, 6}:
            return k_position == 11
        elif q_position in {8, 11}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 6
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {17, 28, 14}:
            return k_position == 15
        elif q_position in {18}:
            return k_position == 13
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21}:
            return k_position == 0
        elif q_position in {22, 23}:
            return k_position == 18
        elif q_position in {24}:
            return k_position == 14
        elif q_position in {26}:
            return k_position == 12
        elif q_position in {27}:
            return k_position == 26

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_attn_1_0_output, k_attn_1_0_output):
        if q_attn_1_0_output in {0, 15, 17, 22, 26, 27}:
            return k_attn_1_0_output == 4
        elif q_attn_1_0_output in {1}:
            return k_attn_1_0_output == 11
        elif q_attn_1_0_output in {2, 16, 21, 24, 25}:
            return k_attn_1_0_output == 5
        elif q_attn_1_0_output in {3}:
            return k_attn_1_0_output == 13
        elif q_attn_1_0_output in {4}:
            return k_attn_1_0_output == 21
        elif q_attn_1_0_output in {10, 12, 5, 7}:
            return k_attn_1_0_output == 9
        elif q_attn_1_0_output in {9, 6}:
            return k_attn_1_0_output == 7
        elif q_attn_1_0_output in {8, 19}:
            return k_attn_1_0_output == 8
        elif q_attn_1_0_output in {11, 14, 18, 23, 28}:
            return k_attn_1_0_output == 6
        elif q_attn_1_0_output in {13}:
            return k_attn_1_0_output == 10
        elif q_attn_1_0_output in {20}:
            return k_attn_1_0_output == 23
        elif q_attn_1_0_output in {29}:
            return k_attn_1_0_output == 28

    attn_2_1_pattern = select_closest(attn_1_0_outputs, attn_1_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_1_2_output, token):
        if attn_1_2_output in {
            0,
            1,
            2,
            4,
            5,
            8,
            9,
            10,
            11,
            12,
            21,
            22,
            24,
            25,
            26,
            27,
            29,
        }:
            return token == "4"
        elif attn_1_2_output in {19, 17, 3, 20}:
            return token == "3"
        elif attn_1_2_output in {6, 7, 13, 15, 16}:
            return token == "2"
        elif attn_1_2_output in {14}:
            return token == "0"
        elif attn_1_2_output in {18}:
            return token == "5"
        elif attn_1_2_output in {28, 23}:
            return token == ""

    attn_2_2_pattern = select_closest(tokens, attn_1_2_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"4", "2", "<s>"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "<s>"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_1_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, attn_0_2_output):
        if attn_1_0_output in {0}:
            return attn_0_2_output == 0
        elif attn_1_0_output in {1}:
            return attn_0_2_output == 3
        elif attn_1_0_output in {2, 10, 13, 6}:
            return attn_0_2_output == 20
        elif attn_1_0_output in {8, 11, 3}:
            return attn_0_2_output == 24
        elif attn_1_0_output in {29, 4, 20, 15}:
            return attn_0_2_output == 21
        elif attn_1_0_output in {5}:
            return attn_0_2_output == 5
        elif attn_1_0_output in {9, 7}:
            return attn_0_2_output == 26
        elif attn_1_0_output in {12, 14}:
            return attn_0_2_output == 23
        elif attn_1_0_output in {16, 18, 26}:
            return attn_0_2_output == 29
        elif attn_1_0_output in {17}:
            return attn_0_2_output == 8
        elif attn_1_0_output in {19}:
            return attn_0_2_output == 9
        elif attn_1_0_output in {21, 23}:
            return attn_0_2_output == 27
        elif attn_1_0_output in {24, 25, 22}:
            return attn_0_2_output == 22
        elif attn_1_0_output in {27}:
            return attn_0_2_output == 28
        elif attn_1_0_output in {28}:
            return attn_0_2_output == 25

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_3_output, attn_0_0_output):
        if attn_1_3_output in {0, 20, 23, 25, 26, 27, 29}:
            return attn_0_0_output == 0
        elif attn_1_3_output in {1}:
            return attn_0_0_output == 1
        elif attn_1_3_output in {2, 22}:
            return attn_0_0_output == 2
        elif attn_1_3_output in {3, 21}:
            return attn_0_0_output == 3
        elif attn_1_3_output in {24, 4, 28, 6}:
            return attn_0_0_output == 4
        elif attn_1_3_output in {5}:
            return attn_0_0_output == 5
        elif attn_1_3_output in {7}:
            return attn_0_0_output == 7
        elif attn_1_3_output in {8}:
            return attn_0_0_output == 8
        elif attn_1_3_output in {9}:
            return attn_0_0_output == 9
        elif attn_1_3_output in {10}:
            return attn_0_0_output == 6
        elif attn_1_3_output in {11, 15}:
            return attn_0_0_output == 11
        elif attn_1_3_output in {12, 13}:
            return attn_0_0_output == 12
        elif attn_1_3_output in {14}:
            return attn_0_0_output == 14
        elif attn_1_3_output in {16}:
            return attn_0_0_output == 15
        elif attn_1_3_output in {17}:
            return attn_0_0_output == 16
        elif attn_1_3_output in {18}:
            return attn_0_0_output == 18
        elif attn_1_3_output in {19}:
            return attn_0_0_output == 19

    num_attn_2_1_pattern = select(attn_0_0_outputs, attn_1_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, position):
        if attn_1_0_output in {0}:
            return position == 5
        elif attn_1_0_output in {1, 2, 3, 6}:
            return position == 15
        elif attn_1_0_output in {4}:
            return position == 14
        elif attn_1_0_output in {5}:
            return position == 11
        elif attn_1_0_output in {7, 8, 9, 20, 21, 29}:
            return position == 16
        elif attn_1_0_output in {10, 11}:
            return position == 18
        elif attn_1_0_output in {27, 12}:
            return position == 21
        elif attn_1_0_output in {13}:
            return position == 20
        elif attn_1_0_output in {17, 14}:
            return position == 23
        elif attn_1_0_output in {16, 26, 15}:
            return position == 19
        elif attn_1_0_output in {18}:
            return position == 26
        elif attn_1_0_output in {19}:
            return position == 0
        elif attn_1_0_output in {22}:
            return position == 17
        elif attn_1_0_output in {24, 23}:
            return position == 29
        elif attn_1_0_output in {25}:
            return position == 27
        elif attn_1_0_output in {28}:
            return position == 25

    num_attn_2_2_pattern = select(positions, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, attn_1_2_output):
        if attn_1_0_output in {0, 25, 5}:
            return attn_1_2_output == 0
        elif attn_1_0_output in {1, 2, 3}:
            return attn_1_2_output == 3
        elif attn_1_0_output in {4}:
            return attn_1_2_output == 4
        elif attn_1_0_output in {6}:
            return attn_1_2_output == 26
        elif attn_1_0_output in {28, 7}:
            return attn_1_2_output == 24
        elif attn_1_0_output in {8}:
            return attn_1_2_output == 23
        elif attn_1_0_output in {16, 9, 27}:
            return attn_1_2_output == 20
        elif attn_1_0_output in {10}:
            return attn_1_2_output == 27
        elif attn_1_0_output in {11, 15, 17, 24, 26}:
            return attn_1_2_output == 22
        elif attn_1_0_output in {18, 12, 20, 22}:
            return attn_1_2_output == 28
        elif attn_1_0_output in {21, 13}:
            return attn_1_2_output == 25
        elif attn_1_0_output in {19, 14}:
            return attn_1_2_output == 21
        elif attn_1_0_output in {29, 23}:
            return attn_1_2_output == 2

    num_attn_2_3_pattern = select(attn_1_2_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_1_output, mlp_0_0_output):
        key = (num_mlp_0_1_output, mlp_0_0_output)
        if key in {
            (1, 0),
            (1, 10),
            (1, 17),
            (1, 21),
            (1, 23),
            (1, 27),
            (1, 29),
            (8, 0),
            (8, 10),
            (8, 17),
            (8, 21),
            (8, 23),
            (8, 29),
            (14, 0),
            (14, 17),
            (14, 21),
            (14, 29),
            (18, 24),
            (18, 27),
            (18, 29),
            (19, 17),
            (20, 17),
            (20, 29),
            (22, 0),
            (22, 17),
            (22, 21),
            (22, 29),
            (23, 0),
            (23, 10),
            (23, 17),
            (23, 21),
            (23, 23),
            (23, 29),
            (26, 0),
            (26, 3),
            (26, 10),
            (26, 17),
            (26, 21),
            (26, 23),
            (26, 26),
            (26, 27),
            (26, 29),
            (29, 17),
        }:
            return 24
        elif key in {
            (3, 27),
            (3, 29),
            (6, 24),
            (6, 27),
            (6, 29),
            (17, 24),
            (17, 27),
            (17, 29),
            (18, 0),
            (18, 2),
            (18, 18),
            (18, 23),
            (18, 26),
            (20, 24),
            (20, 27),
            (24, 24),
            (24, 27),
            (24, 29),
            (25, 24),
            (25, 27),
            (25, 29),
            (29, 24),
            (29, 27),
            (29, 29),
        }:
            return 10
        return 11

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(token, attn_1_1_output):
        key = (token, attn_1_1_output)
        return 1

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(tokens, attn_1_1_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_2_2_output):
        key = (num_attn_2_3_output, num_attn_2_2_output)
        return 2

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_2_0_output):
        key = (num_attn_1_1_output, num_attn_2_0_output)
        return 5

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_2_0_outputs)
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

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
        "output/length/rasp/mostfreq/trainlength20/s1/most_freq_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 12, 13, 14}:
            return token == "5"
        elif position in {1, 9, 10, 15, 16, 17, 18, 19}:
            return token == "4"
        elif position in {8, 2}:
            return token == "0"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {11, 7}:
            return token == "2"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {25, 5}:
            return k_position == 8
        elif q_position in {6}:
            return k_position == 12
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return k_position == 0
        elif q_position in {17, 27}:
            return k_position == 13
        elif q_position in {20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 28
        elif q_position in {26, 22}:
            return k_position == 19
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {28}:
            return k_position == 14
        elif q_position in {29}:
            return k_position == 20

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 15
        elif q_position in {2, 22}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 5, 6, 15, 16, 17, 18}:
            return k_position == 3
        elif q_position in {7, 8, 9, 10, 19}:
            return k_position == 4
        elif q_position in {11, 12, 13, 14}:
            return k_position == 2
        elif q_position in {20}:
            return k_position == 29
        elif q_position in {27, 21}:
            return k_position == 21
        elif q_position in {24, 23}:
            return k_position == 25
        elif q_position in {25}:
            return k_position == 19
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 27

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 11, 12, 13, 15, 18, 19}:
            return token == "5"
        elif position in {1, 2, 7, 8, 9, 10, 17}:
            return token == "1"
        elif position in {3, 4, 5, 6}:
            return token == "0"
        elif position in {16, 24, 14}:
            return token == "4"
        elif position in {20, 21, 22, 23, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 5, 6, 12, 13}:
            return token == "5"
        elif position in {1, 2, 3, 4, 9, 17}:
            return token == "1"
        elif position in {7, 10, 11, 14, 15, 16, 19}:
            return token == "2"
        elif position in {8}:
            return token == "3"
        elif position in {18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 4, 5, 6, 8, 9, 10, 13, 17, 18, 19}:
            return token == "5"
        elif position in {1, 2, 3}:
            return token == "3"
        elif position in {7, 12, 14, 20, 27}:
            return token == "1"
        elif position in {11}:
            return token == "0"
        elif position in {16, 15}:
            return token == "<pad>"
        elif position in {21, 22, 23, 25, 26, 28, 29}:
            return token == ""
        elif position in {24}:
            return token == "4"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 1
        elif q_position in {1, 18, 19, 13}:
            return k_position == 0
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {4, 7}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {16, 10, 14}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {17, 26, 12}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {25, 27, 20}:
            return k_position == 24
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {23}:
            return k_position == 23
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 21
        elif q_position in {29}:
            return k_position == 17

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 9, 11}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 13, 14, 15}:
            return k_position == 0
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {17, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {12, 23}:
            return k_position == 16
        elif q_position in {16, 25}:
            return k_position == 13
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 9
        elif q_position in {20, 28}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 8
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 12
        elif q_position in {27}:
            return k_position == 14
        elif q_position in {29}:
            return k_position == 28

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {8, 16, 7}:
            return token == "4"
        elif position in {9, 10, 11, 12, 13, 14, 15, 17, 18}:
            return token == "3"
        elif position in {19}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 19}:
            return k_position == 1
        elif q_position in {1, 2, 12, 5}:
            return k_position == 0
        elif q_position in {3, 4, 10, 14, 16}:
            return k_position == 4
        elif q_position in {6, 7}:
            return k_position == 3
        elif q_position in {8, 26, 21}:
            return k_position == 8
        elif q_position in {9, 13}:
            return k_position == 6
        elif q_position in {11}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {17, 23}:
            return k_position == 12
        elif q_position in {18}:
            return k_position == 5
        elif q_position in {20}:
            return k_position == 29
        elif q_position in {22}:
            return k_position == 19
        elif q_position in {24}:
            return k_position == 7
        elif q_position in {25, 28}:
            return k_position == 9
        elif q_position in {27}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 22

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1}:
            return token == "2"
        elif position in {2, 3, 4, 5, 6, 8}:
            return token == "<s>"
        elif position in {
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
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
        elif position in {16, 17, 18}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 2}:
            return token == "4"
        elif position in {8, 3, 4, 5}:
            return token == "<s>"
        elif position in {
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            25,
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif position in {24, 22, 15}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 7, 10, 15, 19}:
            return token == "2"
        elif position in {1, 11, 9, 14}:
            return token == "4"
        elif position in {2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {5, 6, 8, 12, 13, 16, 17}:
            return token == "3"
        elif position in {18}:
            return token == "5"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 18}:
            return token == "5"
        elif position in {1}:
            return token == "0"
        elif position in {2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {8, 9, 16, 7}:
            return token == "4"
        elif position in {10, 11, 12, 13, 14, 15, 17}:
            return token == "3"
        elif position in {19}:
            return token == "<s>"
        elif position in {20}:
            return token == "<pad>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 17, 19, 13}:
            return token == "<s>"
        elif position in {8, 1, 12, 16}:
            return token == "3"
        elif position in {2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {9, 11, 14, 7}:
            return token == "4"
        elif position in {10, 15}:
            return token == "2"
        elif position in {18}:
            return token == "5"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 9, 14}:
            return token == "2"
        elif position in {1, 10, 12, 17, 18, 19}:
            return token == "4"
        elif position in {2, 3, 20, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {21, 4, 5}:
            return token == "<pad>"
        elif position in {6}:
            return token == "<s>"
        elif position in {7, 8, 11, 13, 15}:
            return token == "5"
        elif position in {16}:
            return token == "1"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_5_output):
        key = (attn_0_1_output, attn_0_5_output)
        if key in {
            ("0", "3"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "3"),
            ("1", "<s>"),
            ("2", "1"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "1"),
            ("3", "3"),
            ("3", "<s>"),
            ("4", "3"),
            ("4", "<s>"),
            ("<s>", "3"),
            ("<s>", "<s>"),
        }:
            return 9
        elif key in {("1", "2"), ("2", "0"), ("2", "2"), ("3", "0"), ("3", "2")}:
            return 19
        return 28

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_5_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_7_output, attn_0_1_output):
        key = (attn_0_7_output, attn_0_1_output)
        return 24

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_3_output):
        key = attn_0_3_output
        return 7

    mlp_0_2_outputs = [mlp_0_2(k0) for k0 in attn_0_3_outputs]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        return 17

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output):
        key = num_attn_0_4_output
        if key in {0}:
            return 27
        return 3

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
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
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
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
        }:
            return 5
        elif key in {(2, 10), (2, 11), (2, 12), (2, 13), (3, 28), (3, 29)}:
            return 2
        return 29

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_6_output, num_attn_0_5_output):
        key = (num_attn_0_6_output, num_attn_0_5_output)
        return 17

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 3

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 1, 2}:
            return token == "1"
        elif position in {3, 4, 5}:
            return token == "0"
        elif position in {6}:
            return token == "5"
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19}:
            return token == "<s>"
        elif position in {18, 22}:
            return token == "2"
        elif position in {20, 28}:
            return token == "<pad>"
        elif position in {21, 23, 24, 25, 26, 27, 29}:
            return token == ""

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"1", "0", "5", "4"}:
            return position == 4
        elif token in {"2"}:
            return position == 5
        elif token in {"<s>", "3"}:
            return position == 1

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, num_mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"2", "5", "<s>"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "<s>"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 6, 20, 21, 23, 24, 25, 26, 28}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2, 7, 8, 9, 10, 11, 12, 15}:
            return k_position == 1
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {13, 14, 16, 17, 18, 19}:
            return k_position == 19
        elif q_position in {22}:
            return k_position == 16
        elif q_position in {27, 29}:
            return k_position == 17

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_position, k_position):
        if q_position in {0, 23}:
            return k_position == 18
        elif q_position in {1, 2}:
            return k_position == 4
        elif q_position in {3, 11, 12, 13, 17, 24}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 17
        elif q_position in {8, 6, 7}:
            return k_position == 0
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {10}:
            return k_position == 1
        elif q_position in {14}:
            return k_position == 13
        elif q_position in {18, 15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 7
        elif q_position in {19}:
            return k_position == 24
        elif q_position in {20}:
            return k_position == 8
        elif q_position in {29, 21}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 21
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {28}:
            return k_position == 22

    attn_1_4_pattern = select_closest(positions, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, num_mlp_0_1_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_0_output, token):
        if attn_0_0_output in {"0"}:
            return token == "1"
        elif attn_0_0_output in {"1"}:
            return token == "4"
        elif attn_0_0_output in {"2"}:
            return token == "0"
        elif attn_0_0_output in {"<s>", "3", "5"}:
            return token == ""
        elif attn_0_0_output in {"4"}:
            return token == "2"

    attn_1_5_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1}:
            return k_position == 4
        elif q_position in {25, 2}:
            return k_position == 9
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {17, 4, 29, 23}:
            return k_position == 8
        elif q_position in {9, 5}:
            return k_position == 0
        elif q_position in {19, 13, 6, 15}:
            return k_position == 7
        elif q_position in {16, 14, 7}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 21
        elif q_position in {10}:
            return k_position == 27
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {18, 12}:
            return k_position == 6
        elif q_position in {20}:
            return k_position == 26
        elif q_position in {21}:
            return k_position == 20
        elif q_position in {22}:
            return k_position == 18
        elif q_position in {24, 26}:
            return k_position == 3
        elif q_position in {27}:
            return k_position == 24
        elif q_position in {28}:
            return k_position == 11

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_7_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0, 4, 5, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {2}:
            return token == "5"
        elif position in {3}:
            return token == "0"
        elif position in {16, 7}:
            return token == "2"
        elif position in {8, 11, 12, 13, 15, 17, 19}:
            return token == "3"
        elif position in {9, 10, 14}:
            return token == "1"
        elif position in {18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_6_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_6_output, attn_0_1_output):
        if attn_0_6_output in {"<s>", "5", "4", "3", "0", "1"}:
            return attn_0_1_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_0_1_output == "2"

    num_attn_1_0_pattern = select(attn_0_1_outputs, attn_0_6_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_6_output, attn_0_1_output):
        if attn_0_6_output in {"<s>", "5", "4", "0", "1"}:
            return attn_0_1_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_0_1_output == "2"
        elif attn_0_6_output in {"3"}:
            return attn_0_1_output == "3"

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_6_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_4_output, mlp_0_0_output):
        if attn_0_4_output in {"0"}:
            return mlp_0_0_output == 2
        elif attn_0_4_output in {"2", "1"}:
            return mlp_0_0_output == 18
        elif attn_0_4_output in {"3"}:
            return mlp_0_0_output == 20
        elif attn_0_4_output in {"4"}:
            return mlp_0_0_output == 15
        elif attn_0_4_output in {"5"}:
            return mlp_0_0_output == 9
        elif attn_0_4_output in {"<s>"}:
            return mlp_0_0_output == 14

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_4_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 8, 9, 10, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28}:
            return token == ""
        elif position in {4, 5, 6}:
            return token == "2"
        elif position in {7, 11, 12, 15, 16, 17, 18, 19, 20, 29}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_6_output, attn_0_1_output):
        if attn_0_6_output in {"<s>", "5", "3", "2", "0", "1"}:
            return attn_0_1_output == ""
        elif attn_0_6_output in {"4"}:
            return attn_0_1_output == "4"

    num_attn_1_4_pattern = select(attn_0_1_outputs, attn_0_6_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_3_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, token):
        if position in {0, 2, 3}:
            return token == "0"
        elif position in {1}:
            return token == "<s>"
        elif position in {4, 5, 6, 20, 22, 23, 24, 25, 26, 28, 29}:
            return token == ""
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "<pad>"
        elif position in {21}:
            return token == "5"
        elif position in {27}:
            return token == "3"

    num_attn_1_5_pattern = select(tokens, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, ones)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_6_output, token):
        if attn_0_6_output in {"0", "5"}:
            return token == "<pad>"
        elif attn_0_6_output in {"1"}:
            return token == "1"
        elif attn_0_6_output in {"2", "3", "<s>", "4"}:
            return token == ""

    num_attn_1_6_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, token):
        if position in {0, 18, 12}:
            return token == "0"
        elif position in {1, 7}:
            return token == "3"
        elif position in {2, 4, 20, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {3, 21}:
            return token == "<pad>"
        elif position in {13, 5, 6, 14}:
            return token == "4"
        elif position in {8, 9, 16, 15}:
            return token == "<s>"
        elif position in {19, 17, 10, 11}:
            return token == "1"

    num_attn_1_7_pattern = select(tokens, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_7_output, position):
        key = (attn_0_7_output, position)
        if key in {
            ("0", 1),
            ("0", 20),
            ("0", 21),
            ("0", 22),
            ("0", 23),
            ("0", 24),
            ("0", 25),
            ("0", 26),
            ("0", 27),
            ("0", 28),
            ("0", 29),
            ("5", 1),
            ("5", 20),
            ("5", 21),
            ("5", 22),
            ("5", 23),
            ("5", 24),
            ("5", 25),
            ("5", 26),
            ("5", 28),
            ("5", 29),
            ("<s>", 1),
            ("<s>", 29),
        }:
            return 27
        return 5

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, positions)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, num_mlp_0_2_output):
        key = (attn_1_7_output, num_mlp_0_2_output)
        return 1

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_0_6_output, attn_1_5_output):
        key = (attn_0_6_output, attn_1_5_output)
        return 26

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_1_5_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_5_output, attn_1_0_output):
        key = (attn_1_5_output, attn_1_0_output)
        return 7

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_5_outputs, attn_1_0_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_5_output):
        key = (num_attn_1_2_output, num_attn_1_5_output)
        return 3

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_6_output):
        key = (num_attn_1_1_output, num_attn_1_6_output)
        return 1

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_3_output, num_attn_1_1_output):
        key = (num_attn_1_3_output, num_attn_1_1_output)
        return 20

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_6_output, num_attn_1_4_output):
        key = (num_attn_1_6_output, num_attn_1_4_output)
        return 28

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_0_output, position):
        if attn_0_0_output in {"3", "0", "4"}:
            return position == 0
        elif attn_0_0_output in {"2", "1"}:
            return position == 1
        elif attn_0_0_output in {"5"}:
            return position == 9
        elif attn_0_0_output in {"<s>"}:
            return position == 21

    attn_2_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, num_mlp_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_0_output, position):
        if attn_0_0_output in {"0"}:
            return position == 1
        elif attn_0_0_output in {"1"}:
            return position == 0
        elif attn_0_0_output in {"2"}:
            return position == 9
        elif attn_0_0_output in {"3"}:
            return position == 10
        elif attn_0_0_output in {"4"}:
            return position == 7
        elif attn_0_0_output in {"5"}:
            return position == 20
        elif attn_0_0_output in {"<s>"}:
            return position == 27

    attn_2_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_6_output, position):
        if attn_0_6_output in {"<s>", "3", "0"}:
            return position == 0
        elif attn_0_6_output in {"1", "4"}:
            return position == 29
        elif attn_0_6_output in {"2"}:
            return position == 13
        elif attn_0_6_output in {"5"}:
            return position == 6

    attn_2_2_pattern = select_closest(positions, attn_0_6_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_4_output, position):
        if attn_0_4_output in {"<s>", "0", "5", "4"}:
            return position == 7
        elif attn_0_4_output in {"1", "3"}:
            return position == 0
        elif attn_0_4_output in {"2"}:
            return position == 22

    attn_2_3_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, num_mlp_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_6_output, token):
        if attn_0_6_output in {"0"}:
            return token == "2"
        elif attn_0_6_output in {"<s>", "1"}:
            return token == "<s>"
        elif attn_0_6_output in {"2"}:
            return token == "1"
        elif attn_0_6_output in {"3", "5", "4"}:
            return token == ""

    attn_2_4_pattern = select_closest(tokens, attn_0_6_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, tokens)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_3_output, token):
        if attn_0_3_output in {"2", "1", "0"}:
            return token == "4"
        elif attn_0_3_output in {"3"}:
            return token == "<s>"
        elif attn_0_3_output in {"4"}:
            return token == "1"
        elif attn_0_3_output in {"<s>", "5"}:
            return token == "3"

    attn_2_5_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, mlp_1_0_output):
        if attn_0_0_output in {"<s>", "0", "4"}:
            return mlp_1_0_output == 27
        elif attn_0_0_output in {"2", "1"}:
            return mlp_1_0_output == 6
        elif attn_0_0_output in {"3"}:
            return mlp_1_0_output == 22
        elif attn_0_0_output in {"5"}:
            return mlp_1_0_output == 2

    attn_2_6_pattern = select_closest(mlp_1_0_outputs, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"5", "4", "3", "2", "1"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_7_pattern = select_closest(tokens, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_2_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(position, token):
        if position in {
            0,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
            11,
            16,
            17,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            28,
            29,
        }:
            return token == ""
        elif position in {1, 2, 27}:
            return token == "3"
        elif position in {7, 12, 13, 14, 15, 18, 22}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, positions, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(position, token):
        if position in {0, 22, 24, 25, 26}:
            return token == "0"
        elif position in {1}:
            return token == "5"
        elif position in {2, 5, 6, 20, 21, 23, 28, 29}:
            return token == ""
        elif position in {3, 4}:
            return token == "<pad>"
        elif position in {8, 16, 7}:
            return token == "4"
        elif position in {9, 10, 11, 13, 18}:
            return token == "3"
        elif position in {12, 14, 15, 17, 19, 27}:
            return token == "<s>"

    num_attn_2_1_pattern = select(tokens, positions, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_3_output):
        if position in {0}:
            return attn_0_3_output == "<pad>"
        elif position in {1, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return attn_0_3_output == ""
        elif position in {3}:
            return attn_0_3_output == "<s>"
        elif position in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return attn_0_3_output == "1"

    num_attn_2_2_pattern = select(attn_0_3_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_6_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(token, position):
        if token in {"1", "0"}:
            return position == 27
        elif token in {"2"}:
            return position == 17
        elif token in {"3", "4"}:
            return position == 5
        elif token in {"5"}:
            return position == 28
        elif token in {"<s>"}:
            return position == 10

    num_attn_2_3_pattern = select(positions, tokens, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"<s>", "5", "2", "0", "3"}:
            return k_attn_0_0_output == "1"
        elif q_attn_0_0_output in {"1"}:
            return k_attn_0_0_output == "0"
        elif q_attn_0_0_output in {"4"}:
            return k_attn_0_0_output == ""

    num_attn_2_4_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_6_output, attn_1_6_output):
        if attn_0_6_output in {"0"}:
            return attn_1_6_output == "0"
        elif attn_0_6_output in {"<s>", "5", "4", "3", "2", "1"}:
            return attn_1_6_output == ""

    num_attn_2_5_pattern = select(attn_1_6_outputs, attn_0_6_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_1_5_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 8
        elif position in {1}:
            return mlp_0_0_output == 19
        elif position in {2}:
            return mlp_0_0_output == 9
        elif position in {3, 28, 22}:
            return mlp_0_0_output == 14
        elif position in {4}:
            return mlp_0_0_output == 18
        elif position in {5, 6}:
            return mlp_0_0_output == 28
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return mlp_0_0_output == 0
        elif position in {20, 23, 24, 26, 29}:
            return mlp_0_0_output == 20
        elif position in {21}:
            return mlp_0_0_output == 7
        elif position in {25, 27}:
            return mlp_0_0_output == 29

    num_attn_2_6_pattern = select(mlp_0_0_outputs, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, ones)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"<s>", "5", "4", "3", "2", "1"}:
            return token == ""

    num_attn_2_7_pattern = select(tokens, attn_0_3_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_1_output, mlp_1_2_output):
        key = (num_mlp_0_1_output, mlp_1_2_output)
        return 0

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, mlp_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_1_3_output, mlp_1_3_output):
        key = (num_mlp_1_3_output, mlp_1_3_output)
        return 10

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_1_3_outputs, mlp_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_0_1_output, mlp_1_3_output):
        key = (attn_0_1_output, mlp_1_3_output)
        return 6

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_0_1_outputs, mlp_1_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_0_3_output, attn_2_4_output):
        key = (num_mlp_0_3_output, attn_2_4_output)
        return 15

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, attn_2_4_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_5_output, num_attn_2_2_output):
        key = (num_attn_0_5_output, num_attn_2_2_output)
        if key in {
            (0, 79),
            (0, 80),
            (0, 81),
            (0, 82),
            (0, 83),
            (0, 84),
            (0, 85),
            (0, 86),
            (0, 87),
            (0, 88),
            (0, 89),
        }:
            return 24
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_7_output, num_attn_1_6_output):
        key = (num_attn_2_7_output, num_attn_1_6_output)
        return 22

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_1_output, num_attn_0_5_output):
        key = (num_attn_1_1_output, num_attn_0_5_output)
        return 3

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_2_2_output, num_attn_2_4_output):
        key = (num_attn_2_2_output, num_attn_2_4_output)
        return 28

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
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

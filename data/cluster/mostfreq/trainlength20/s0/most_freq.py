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
        "output/length/rasp/mostfreq/trainlength20/s0/most_freq_weights.csv",
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
        if q_position in {0, 8, 10, 13, 15, 16}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {4, 7}:
            return k_position == 5
        elif q_position in {5, 14}:
            return k_position == 6
        elif q_position in {12, 6}:
            return k_position == 9
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {11, 21}:
            return k_position == 19
        elif q_position in {17}:
            return k_position == 12
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 2
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 23
        elif q_position in {23}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 18
        elif q_position in {25, 26, 29}:
            return k_position == 28
        elif q_position in {27}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 11

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 3}:
            return k_position == 1
        elif q_position in {2, 11, 13}:
            return k_position == 3
        elif q_position in {10, 4, 7}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {9, 6}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {16, 14}:
            return k_position == 9
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {17, 18}:
            return k_position == 12
        elif q_position in {19}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 16
        elif q_position in {21}:
            return k_position == 15
        elif q_position in {28, 22, 23}:
            return k_position == 24
        elif q_position in {24}:
            return k_position == 27
        elif q_position in {25}:
            return k_position == 22
        elif q_position in {26}:
            return k_position == 20
        elif q_position in {27, 29}:
            return k_position == 25

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 18, 11}:
            return token == "0"
        elif position in {1, 2}:
            return token == "3"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {7, 9, 10, 15, 24}:
            return token == "4"
        elif position in {8, 12, 13, 14, 16, 19}:
            return token == "2"
        elif position in {17}:
            return token == "5"
        elif position in {20, 21, 22, 23, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 9, 18, 14}:
            return token == "5"
        elif position in {1, 3, 4, 5, 8, 11, 15, 17, 20, 25}:
            return token == "2"
        elif position in {2, 10, 12}:
            return token == "0"
        elif position in {6}:
            return token == "1"
        elif position in {7}:
            return token == "3"
        elif position in {16, 19, 13}:
            return token == "4"
        elif position in {21, 22, 23, 24, 26, 27, 28, 29}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 7, 9, 12, 18, 22}:
            return token == "4"
        elif position in {1}:
            return token == "5"
        elif position in {2}:
            return token == "1"
        elif position in {3, 4, 5, 6, 13, 17}:
            return token == "3"
        elif position in {8, 10, 11, 14, 15}:
            return token == "0"
        elif position in {16, 20, 21, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {19}:
            return token == "2"

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 14, 15}:
            return token == "0"
        elif position in {2, 7, 16, 17, 18, 19, 21, 22, 26, 28}:
            return token == ""
        elif position in {3, 5}:
            return token == "2"
        elif position in {4, 10, 11, 12, 13}:
            return token == "3"
        elif position in {6}:
            return token == "5"
        elif position in {8, 20, 24, 25, 27, 29}:
            return token == "1"
        elif position in {9}:
            return token == "4"
        elif position in {23}:
            return token == "<pad>"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2, 3}:
            return k_position == 1
        elif q_position in {1, 10, 6}:
            return k_position == 4
        elif q_position in {4, 13, 7}:
            return k_position == 8
        elif q_position in {19, 5}:
            return k_position == 2
        elif q_position in {8}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 18
        elif q_position in {12}:
            return k_position == 6
        elif q_position in {27, 14, 23}:
            return k_position == 12
        elif q_position in {20, 15}:
            return k_position == 11
        elif q_position in {16, 24}:
            return k_position == 20
        elif q_position in {17}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {21}:
            return k_position == 9
        elif q_position in {22}:
            return k_position == 21
        elif q_position in {25}:
            return k_position == 13
        elif q_position in {26}:
            return k_position == 24
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 27

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 15, 16, 17, 19}:
            return token == "0"
        elif position in {8, 1, 12, 14}:
            return token == "2"
        elif position in {18, 2, 10, 13}:
            return token == "4"
        elif position in {3, 4, 5, 6, 7, 9, 11}:
            return token == "5"
        elif position in {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 25}:
            return token == "<pad>"
        elif position in {1}:
            return token == "4"
        elif position in {2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 26, 27, 28, 29}:
            return token == ""
        elif position in {8, 7}:
            return token == "3"
        elif position in {9, 10}:
            return token == "0"
        elif position in {11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "2"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {
            0,
            2,
            3,
            4,
            9,
            10,
            12,
            13,
            15,
            16,
            17,
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
        elif position in {1, 11}:
            return token == "2"
        elif position in {5, 6}:
            return token == "5"
        elif position in {8, 7}:
            return token == "<s>"
        elif position in {18, 19, 14}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 8, 9, 7}:
            return k_position == 0
        elif q_position in {1, 2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {18, 11, 6, 15}:
            return k_position == 8
        elif q_position in {24, 10}:
            return k_position == 10
        elif q_position in {12, 14}:
            return k_position == 9
        elif q_position in {25, 29, 13}:
            return k_position == 11
        elif q_position in {16, 19}:
            return k_position == 13
        elif q_position in {17}:
            return k_position == 15
        elif q_position in {20}:
            return k_position == 28
        elif q_position in {21}:
            return k_position == 21
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {28, 23}:
            return k_position == 16
        elif q_position in {26}:
            return k_position == 14
        elif q_position in {27}:
            return k_position == 26

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 8, 7}:
            return token == "3"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
        }:
            return token == ""
        elif position in {16, 22}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "5"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
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

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 2, 3, 4, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {18, 5, 6}:
            return token == "2"
        elif position in {7, 8, 10, 11, 12, 14, 16, 17}:
            return token == "4"
        elif position in {9}:
            return token == "0"
        elif position in {19, 13}:
            return token == "<s>"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 2, 5, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {
            3,
            4,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
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
        elif position in {18}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1}:
            return token == "1"
        elif position in {
            2,
            3,
            4,
            5,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
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
        elif position in {8, 6, 7}:
            return token == "<s>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position):
        key = position
        return 3

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in positions]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, position):
        key = (attn_0_4_output, position)
        if key in {
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 16),
            ("1", 8),
            ("2", 8),
            ("2", 11),
            ("2", 12),
            ("2", 16),
            ("3", 8),
            ("3", 11),
            ("3", 12),
            ("3", 16),
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 16),
            ("4", 18),
            ("4", 19),
            ("4", 20),
            ("4", 24),
            ("4", 25),
            ("4", 28),
            ("5", 8),
            ("5", 9),
            ("5", 11),
            ("5", 12),
            ("5", 13),
            ("5", 16),
            ("<s>", 8),
            ("<s>", 11),
            ("<s>", 12),
            ("<s>", 16),
        }:
            return 15
        elif key in {
            ("0", 19),
            ("1", 19),
            ("2", 13),
            ("2", 19),
            ("3", 13),
            ("3", 19),
            ("5", 19),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 2),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 9),
            ("<s>", 13),
            ("<s>", 15),
            ("<s>", 17),
            ("<s>", 18),
            ("<s>", 19),
            ("<s>", 20),
            ("<s>", 21),
            ("<s>", 22),
            ("<s>", 23),
            ("<s>", 24),
            ("<s>", 25),
            ("<s>", 26),
            ("<s>", 27),
            ("<s>", 28),
            ("<s>", 29),
        }:
            return 19
        elif key in {
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 14),
            ("0", 15),
            ("0", 17),
            ("0", 18),
            ("1", 9),
            ("2", 18),
            ("4", 9),
            ("4", 10),
            ("4", 14),
            ("4", 15),
            ("4", 17),
            ("4", 21),
            ("4", 22),
            ("4", 23),
            ("4", 26),
            ("4", 27),
            ("4", 29),
            ("5", 10),
            ("5", 17),
            ("5", 18),
            ("<s>", 10),
            ("<s>", 14),
        }:
            return 3
        elif key in {
            ("0", 22),
            ("0", 23),
            ("0", 25),
            ("0", 26),
            ("0", 29),
            ("1", 10),
            ("1", 17),
            ("2", 17),
            ("3", 17),
            ("3", 18),
        }:
            return 9
        elif key in {
            ("1", 11),
            ("1", 12),
            ("1", 13),
            ("1", 14),
            ("1", 16),
            ("1", 18),
            ("5", 14),
        }:
            return 6
        return 13

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output, attn_0_3_output):
        key = (attn_0_6_output, attn_0_3_output)
        if key in {
            ("0", "2"),
            ("1", "2"),
            ("2", "2"),
            ("3", "2"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("<s>", "2"),
        }:
            return 5
        elif key in {
            ("2", "0"),
            ("2", "1"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "<s>"),
            ("4", "1"),
            ("4", "4"),
            ("4", "5"),
            ("5", "2"),
        }:
            return 6
        return 28

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_3_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 17),
            ("0", 18),
            ("0", 19),
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
        }:
            return 17
        elif key in {
            ("0", 2),
            ("0", 3),
            ("0", 9),
            ("1", 2),
            ("1", 3),
            ("2", 2),
            ("2", 3),
            ("3", 2),
            ("3", 3),
            ("4", 2),
            ("4", 3),
            ("5", 2),
            ("5", 3),
            ("<s>", 2),
            ("<s>", 3),
        }:
            return 16
        elif key in {
            ("1", 0),
            ("1", 1),
            ("2", 0),
            ("2", 1),
            ("3", 0),
            ("3", 1),
            ("4", 0),
            ("4", 1),
            ("5", 0),
            ("5", 1),
            ("<s>", 0),
            ("<s>", 1),
        }:
            return 22
        return 10

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output):
        key = num_attn_0_4_output
        return 5

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 5

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 5

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_4_output, num_attn_0_3_output):
        key = (num_attn_0_4_output, num_attn_0_3_output)
        return 1

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"1", "0", "4", "<s>", "2", "5"}:
            return position == 3
        elif token in {"3"}:
            return position == 5

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, attn_0_6_output):
        if token in {"0", "<s>"}:
            return attn_0_6_output == ""
        elif token in {"1", "3", "5"}:
            return attn_0_6_output == "0"
        elif token in {"2"}:
            return attn_0_6_output == "<s>"
        elif token in {"4"}:
            return attn_0_6_output == "2"

    attn_1_1_pattern = select_closest(attn_0_6_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, num_mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0, 18, 13, 15}:
            return k_position == 1
        elif q_position in {1, 19, 4, 14}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 14
        elif q_position in {3}:
            return k_position == 5
        elif q_position in {25, 5}:
            return k_position == 11
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 29, 22}:
            return k_position == 8
        elif q_position in {9, 17, 20, 21, 26}:
            return k_position == 0
        elif q_position in {10}:
            return k_position == 16
        elif q_position in {16, 11, 12}:
            return k_position == 9
        elif q_position in {24, 27, 23}:
            return k_position == 28
        elif q_position in {28}:
            return k_position == 7

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_5_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, mlp_0_3_output):
        if token in {"0"}:
            return mlp_0_3_output == 10
        elif token in {"1", "3", "<s>"}:
            return mlp_0_3_output == 21
        elif token in {"2", "4", "5"}:
            return mlp_0_3_output == 9

    attn_1_3_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_0_output, mlp_0_3_output):
        if attn_0_0_output in {"0"}:
            return mlp_0_3_output == 28
        elif attn_0_0_output in {"1", "3", "<s>", "2", "5"}:
            return mlp_0_3_output == 21
        elif attn_0_0_output in {"4"}:
            return mlp_0_3_output == 17

    attn_1_4_pattern = select_closest(mlp_0_3_outputs, attn_0_0_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_3_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, mlp_0_3_output):
        if token in {"0"}:
            return mlp_0_3_output == 28
        elif token in {"1", "<s>"}:
            return mlp_0_3_output == 21
        elif token in {"2"}:
            return mlp_0_3_output == 10
        elif token in {"3", "5", "4"}:
            return mlp_0_3_output == 22

    attn_1_5_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_2_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, attn_0_6_output):
        if token in {"0", "4", "1"}:
            return attn_0_6_output == "3"
        elif token in {"2", "<s>"}:
            return attn_0_6_output == "0"
        elif token in {"3"}:
            return attn_0_6_output == "4"
        elif token in {"5"}:
            return attn_0_6_output == "<s>"

    attn_1_6_pattern = select_closest(attn_0_6_outputs, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, mlp_0_3_output):
        if position in {0}:
            return mlp_0_3_output == 23
        elif position in {8, 1, 9, 7}:
            return mlp_0_3_output == 10
        elif position in {2, 28}:
            return mlp_0_3_output == 7
        elif position in {3, 4, 5, 6, 20}:
            return mlp_0_3_output == 16
        elif position in {10, 26}:
            return mlp_0_3_output == 12
        elif position in {16, 11}:
            return mlp_0_3_output == 8
        elif position in {12, 15}:
            return mlp_0_3_output == 13
        elif position in {13}:
            return mlp_0_3_output == 0
        elif position in {14}:
            return mlp_0_3_output == 6
        elif position in {17}:
            return mlp_0_3_output == 17
        elif position in {18, 27}:
            return mlp_0_3_output == 9
        elif position in {19}:
            return mlp_0_3_output == 11
        elif position in {21}:
            return mlp_0_3_output == 22
        elif position in {22}:
            return mlp_0_3_output == 28
        elif position in {29, 23}:
            return mlp_0_3_output == 15
        elif position in {24}:
            return mlp_0_3_output == 1
        elif position in {25}:
            return mlp_0_3_output == 19

    attn_1_7_pattern = select_closest(mlp_0_3_outputs, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_1_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(token, attn_0_5_output):
        if token in {"0"}:
            return attn_0_5_output == "3"
        elif token in {"1", "3", "4", "2", "5"}:
            return attn_0_5_output == ""
        elif token in {"<s>"}:
            return attn_0_5_output == "4"

    num_attn_1_0_pattern = select(attn_0_5_outputs, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {
            0,
            2,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
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
        }:
            return token == ""
        elif position in {1}:
            return token == "5"
        elif position in {3}:
            return token == "<s>"
        elif position in {29, 4, 5, 6}:
            return token == "1"
        elif position in {7}:
            return token == "4"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_3_output, attn_0_1_output):
        if mlp_0_3_output in {
            0,
            2,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            15,
            16,
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
        }:
            return attn_0_1_output == ""
        elif mlp_0_3_output in {1, 3, 14}:
            return attn_0_1_output == "<pad>"
        elif mlp_0_3_output in {10, 4, 29}:
            return attn_0_1_output == "3"

    num_attn_1_2_pattern = select(attn_0_1_outputs, mlp_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "0"
        elif position in {24, 1}:
            return token == "1"
        elif position in {2, 12, 14}:
            return token == "<pad>"
        elif position in {3, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 23, 27, 28}:
            return token == ""
        elif position in {4, 5, 6, 10, 25, 29}:
            return token == "2"
        elif position in {26, 22}:
            return token == "3"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(mlp_0_3_output, token):
        if mlp_0_3_output in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            11,
            13,
            14,
            15,
            16,
            18,
            19,
            20,
            21,
            23,
            24,
            26,
            27,
            28,
        }:
            return token == ""
        elif mlp_0_3_output in {17, 12, 5, 25}:
            return token == "<s>"
        elif mlp_0_3_output in {10, 29}:
            return token == "4"
        elif mlp_0_3_output in {22}:
            return token == "<pad>"

    num_attn_1_4_pattern = select(tokens, mlp_0_3_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_2_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(token, attn_0_2_output):
        if token in {"0", "3", "4", "<s>", "2", "5"}:
            return attn_0_2_output == ""
        elif token in {"1"}:
            return attn_0_2_output == "1"

    num_attn_1_5_pattern = select(attn_0_2_outputs, tokens, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_7_output, token):
        if attn_0_7_output in {"1", "0", "3", "4", "<s>"}:
            return token == ""
        elif attn_0_7_output in {"2"}:
            return token == "<pad>"
        elif attn_0_7_output in {"5"}:
            return token == "5"

    num_attn_1_6_pattern = select(tokens, attn_0_7_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_4_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_2_output):
        if position in {0, 1, 2, 3, 24}:
            return attn_0_2_output == "0"
        elif position in {
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
        }:
            return attn_0_2_output == ""
        elif position in {12, 29, 14}:
            return attn_0_2_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_2_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_7_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_2_output, attn_1_5_output):
        key = (num_mlp_0_2_output, attn_1_5_output)
        if key in {
            (0, 29),
            (2, 29),
            (7, 29),
            (10, 29),
            (16, 29),
            (24, 29),
            (27, 29),
            (28, 29),
        }:
            return 6
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(token, attn_1_2_output):
        key = (token, attn_1_2_output)
        return 2

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(tokens, attn_1_2_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_2_output, mlp_0_0_output):
        key = (mlp_0_2_output, mlp_0_0_output)
        if key in {
            (0, 17),
            (2, 17),
            (4, 17),
            (5, 17),
            (8, 17),
            (11, 17),
            (17, 17),
            (20, 17),
            (21, 17),
            (22, 17),
            (24, 17),
        }:
            return 16
        elif key in {(6, 17), (10, 17), (21, 10)}:
            return 8
        return 10

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_0_0_output, attn_0_6_output):
        key = (attn_0_0_output, attn_0_6_output)
        return 4

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_6_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output, num_attn_1_0_output):
        key = (num_attn_0_2_output, num_attn_1_0_output)
        return 20

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 1

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_0_output):
        key = num_attn_1_0_output
        return 9

    num_mlp_1_2_outputs = [num_mlp_1_2(k0) for k0 in num_attn_1_0_outputs]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_3_output):
        key = num_attn_1_3_output
        if key in {0}:
            return 28
        return 6

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, mlp_0_2_output):
        if position in {0, 22}:
            return mlp_0_2_output == 16
        elif position in {1, 28}:
            return mlp_0_2_output == 10
        elif position in {2}:
            return mlp_0_2_output == 28
        elif position in {3}:
            return mlp_0_2_output == 2
        elif position in {20, 27, 4, 5}:
            return mlp_0_2_output == 6
        elif position in {26, 29, 6}:
            return mlp_0_2_output == 15
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return mlp_0_2_output == 5
        elif position in {21}:
            return mlp_0_2_output == 7
        elif position in {23}:
            return mlp_0_2_output == 12
        elif position in {24}:
            return mlp_0_2_output == 19
        elif position in {25}:
            return mlp_0_2_output == 9

    attn_2_0_pattern = select_closest(mlp_0_2_outputs, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_0_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return k_position == 1
        elif q_position in {1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {4, 5, 6}:
            return k_position == 0

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, token):
        if position in {0, 1, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}:
            return token == "0"
        elif position in {2, 3}:
            return token == "4"
        elif position in {4}:
            return token == "5"
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            return token == "<s>"

    attn_2_2_pattern = select_closest(tokens, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, tokens)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_1_output, position):
        if attn_0_1_output in {"0", "3", "<s>"}:
            return position == 3
        elif attn_0_1_output in {"1", "4", "5"}:
            return position == 0
        elif attn_0_1_output in {"2"}:
            return position == 2

    attn_2_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"0", "4"}:
            return k_token == "3"
        elif q_token in {"1", "3"}:
            return k_token == "0"
        elif q_token in {"2", "<s>"}:
            return k_token == ""
        elif q_token in {"5"}:
            return k_token == "4"

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_3_output, mlp_0_3_output):
        if attn_0_3_output in {"0", "<s>"}:
            return mlp_0_3_output == 7
        elif attn_0_3_output in {"1", "5"}:
            return mlp_0_3_output == 10
        elif attn_0_3_output in {"2"}:
            return mlp_0_3_output == 12
        elif attn_0_3_output in {"3"}:
            return mlp_0_3_output == 21
        elif attn_0_3_output in {"4"}:
            return mlp_0_3_output == 22

    attn_2_5_pattern = select_closest(mlp_0_3_outputs, attn_0_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_1_output, mlp_0_3_output):
        if attn_0_1_output in {"0"}:
            return mlp_0_3_output == 10
        elif attn_0_1_output in {"1"}:
            return mlp_0_3_output == 7
        elif attn_0_1_output in {"2"}:
            return mlp_0_3_output == 24
        elif attn_0_1_output in {"3", "5"}:
            return mlp_0_3_output == 21
        elif attn_0_1_output in {"4"}:
            return mlp_0_3_output == 22
        elif attn_0_1_output in {"<s>"}:
            return mlp_0_3_output == 16

    attn_2_6_pattern = select_closest(mlp_0_3_outputs, attn_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_7_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_position, k_position):
        if q_position in {0, 1, 6, 22, 25, 26, 27}:
            return k_position == 0
        elif q_position in {2, 3}:
            return k_position == 16
        elif q_position in {4, 5, 20, 21, 28, 29}:
            return k_position == 7
        elif q_position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19}:
            return k_position == 1
        elif q_position in {17}:
            return k_position == 2
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {24}:
            return k_position == 3

    attn_2_7_pattern = select_closest(positions, positions, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, mlp_0_1_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(mlp_0_3_output, attn_1_2_output):
        if mlp_0_3_output in {
            0,
            1,
            2,
            4,
            5,
            6,
            8,
            10,
            11,
            12,
            13,
            15,
            16,
            20,
            26,
            27,
            28,
            29,
        }:
            return attn_1_2_output == "0"
        elif mlp_0_3_output in {3}:
            return attn_1_2_output == "3"
        elif mlp_0_3_output in {7, 9, 14, 17, 18, 21, 23, 24, 25}:
            return attn_1_2_output == ""
        elif mlp_0_3_output in {19}:
            return attn_1_2_output == "1"
        elif mlp_0_3_output in {22}:
            return attn_1_2_output == "<pad>"

    num_attn_2_0_pattern = select(attn_1_2_outputs, mlp_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_0_output, attn_0_7_output):
        if attn_0_0_output in {"0", "4", "1"}:
            return attn_0_7_output == "<pad>"
        elif attn_0_0_output in {"2"}:
            return attn_0_7_output == "2"
        elif attn_0_0_output in {"3", "<s>"}:
            return attn_0_7_output == ""
        elif attn_0_0_output in {"5"}:
            return attn_0_7_output == "<s>"

    num_attn_2_1_pattern = select(attn_0_7_outputs, attn_0_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(position, attn_0_1_output):
        if position in {
            0,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            17,
            18,
            20,
            21,
            22,
            23,
            25,
            26,
            27,
            28,
        }:
            return attn_0_1_output == ""
        elif position in {24, 1}:
            return attn_0_1_output == "4"
        elif position in {2}:
            return attn_0_1_output == "<s>"
        elif position in {16, 19, 12}:
            return attn_0_1_output == "<pad>"
        elif position in {29}:
            return attn_0_1_output == "2"

    num_attn_2_2_pattern = select(attn_0_1_outputs, positions, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, attn_0_4_output):
        if attn_0_0_output in {"0", "3", "4", "2", "5"}:
            return attn_0_4_output == ""
        elif attn_0_0_output in {"1"}:
            return attn_0_4_output == "1"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_4_output == "<pad>"

    num_attn_2_3_pattern = select(attn_0_4_outputs, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_1_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(token, position):
        if token in {"0", "3", "5"}:
            return position == 7
        elif token in {"1"}:
            return position == 14
        elif token in {"2"}:
            return position == 17
        elif token in {"4"}:
            return position == 21
        elif token in {"<s>"}:
            return position == 24

    num_attn_2_4_pattern = select(positions, tokens, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_1_1_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(mlp_0_3_output, attn_0_6_output):
        if mlp_0_3_output in {
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            26,
            27,
            28,
            29,
        }:
            return attn_0_6_output == ""
        elif mlp_0_3_output in {25, 10, 3, 12}:
            return attn_0_6_output == "2"

    num_attn_2_5_pattern = select(attn_0_6_outputs, mlp_0_3_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_6_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_4_output, attn_1_2_output):
        if attn_0_4_output in {"0", "3", "<s>"}:
            return attn_1_2_output == ""
        elif attn_0_4_output in {"1", "4", "2", "5"}:
            return attn_1_2_output == "0"

    num_attn_2_6_pattern = select(attn_1_2_outputs, attn_0_4_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_4_output, mlp_1_1_output):
        if attn_1_4_output in {"0"}:
            return mlp_1_1_output == 12
        elif attn_1_4_output in {"1"}:
            return mlp_1_1_output == 7
        elif attn_1_4_output in {"2"}:
            return mlp_1_1_output == 16
        elif attn_1_4_output in {"3"}:
            return mlp_1_1_output == 25
        elif attn_1_4_output in {"4"}:
            return mlp_1_1_output == 15
        elif attn_1_4_output in {"5"}:
            return mlp_1_1_output == 18
        elif attn_1_4_output in {"<s>"}:
            return mlp_1_1_output == 23

    num_attn_2_7_pattern = select(mlp_1_1_outputs, attn_1_4_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_0_output, num_mlp_0_0_output):
        key = (mlp_0_0_output, num_mlp_0_0_output)
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_0_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_2_output, attn_1_7_output):
        key = (attn_0_2_output, attn_1_7_output)
        if key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "2"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "5"),
            ("3", "<s>"),
            ("5", "0"),
            ("5", "1"),
            ("<s>", "1"),
        }:
            return 26
        elif key in {("1", "4"), ("2", "4"), ("3", "4"), ("4", "1")}:
            return 13
        return 14

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_0_7_output, mlp_0_3_output):
        key = (attn_0_7_output, mlp_0_3_output)
        return 14

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_0_7_outputs, mlp_0_3_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_1_2_output, attn_0_4_output):
        key = (attn_1_2_output, attn_0_4_output)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "3"),
            ("5", "3"),
            ("<s>", "3"),
        }:
            return 24
        return 29

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_0_4_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_1_4_output):
        key = (num_attn_2_3_output, num_attn_1_4_output)
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 17

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_3_output, num_attn_2_1_output):
        key = (num_attn_2_3_output, num_attn_2_1_output)
        return 20

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_5_output, num_attn_2_4_output):
        key = (num_attn_1_5_output, num_attn_2_4_output)
        return 12

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_2_4_outputs)
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


print(run(["<s>", "5", "0", "3", "3", "3", "1", "3", "5", "2", "4", "0", "0", "4"]))

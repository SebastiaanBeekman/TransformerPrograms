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
        "output/length/rasp/mostfreq/trainlength30/s2/most_freq_weights.csv",
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
        if q_position in {0, 28, 12, 20}:
            return k_position == 3
        elif q_position in {1, 7}:
            return k_position == 1
        elif q_position in {2, 19}:
            return k_position == 2
        elif q_position in {3, 36, 37, 38}:
            return k_position == 26
        elif q_position in {9, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 27
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {8}:
            return k_position == 4
        elif q_position in {17, 10}:
            return k_position == 7
        elif q_position in {27, 11, 31}:
            return k_position == 8
        elif q_position in {13}:
            return k_position == 17
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 15
        elif q_position in {16}:
            return k_position == 9
        elif q_position in {18, 22}:
            return k_position == 10
        elif q_position in {21}:
            return k_position == 18
        elif q_position in {23}:
            return k_position == 23
        elif q_position in {24}:
            return k_position == 19
        elif q_position in {25}:
            return k_position == 6
        elif q_position in {26}:
            return k_position == 16
        elif q_position in {29}:
            return k_position == 22
        elif q_position in {30}:
            return k_position == 21
        elif q_position in {32}:
            return k_position == 0
        elif q_position in {33}:
            return k_position == 20
        elif q_position in {34}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 24
        elif q_position in {39}:
            return k_position == 30

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {8, 2, 22}:
            return k_position == 4
        elif q_position in {3, 4}:
            return k_position == 1
        elif q_position in {5, 6, 37, 15, 25, 28}:
            return k_position == 3
        elif q_position in {9, 18, 7}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 16
        elif q_position in {12}:
            return k_position == 29
        elif q_position in {13}:
            return k_position == 15
        elif q_position in {14}:
            return k_position == 17
        elif q_position in {16, 19}:
            return k_position == 7
        elif q_position in {17}:
            return k_position == 13
        elif q_position in {20, 38, 31}:
            return k_position == 18
        elif q_position in {27, 21}:
            return k_position == 2
        elif q_position in {23}:
            return k_position == 8
        elif q_position in {24, 36}:
            return k_position == 10
        elif q_position in {26}:
            return k_position == 22
        elif q_position in {29}:
            return k_position == 20
        elif q_position in {30}:
            return k_position == 28
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {33, 39}:
            return k_position == 36
        elif q_position in {34}:
            return k_position == 37
        elif q_position in {35}:
            return k_position == 38

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 12
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3}:
            return k_position == 4
        elif q_position in {35, 4}:
            return k_position == 8
        elif q_position in {5, 38}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 19
        elif q_position in {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}:
            return k_position == 3
        elif q_position in {21, 22, 23, 24, 25, 26, 28, 29}:
            return k_position == 21
        elif q_position in {27}:
            return k_position == 17
        elif q_position in {32, 34, 30}:
            return k_position == 39
        elif q_position in {39, 31}:
            return k_position == 37
        elif q_position in {33}:
            return k_position == 35
        elif q_position in {36}:
            return k_position == 27
        elif q_position in {37}:
            return k_position == 25

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 1, 2, 3, 7, 9, 10, 13, 14, 17, 19, 20, 21, 25, 27}:
            return token == "3"
        elif position in {4, 5, 11, 18, 23}:
            return token == "4"
        elif position in {8, 16, 6}:
            return token == "2"
        elif position in {32, 33, 34, 35, 36, 38, 39, 12, 15, 22, 24, 28, 30, 31}:
            return token == ""
        elif position in {26, 29}:
            return token == "1"
        elif position in {37}:
            return token == "0"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 13}:
            return k_position == 2
        elif q_position in {1, 2, 3, 29}:
            return k_position == 1
        elif q_position in {10, 4, 5}:
            return k_position == 6
        elif q_position in {11, 6}:
            return k_position == 12
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 27}:
            return k_position == 5
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {14}:
            return k_position == 7
        elif q_position in {34, 15}:
            return k_position == 21
        elif q_position in {16}:
            return k_position == 11
        elif q_position in {17, 26}:
            return k_position == 14
        elif q_position in {18, 23}:
            return k_position == 13
        elif q_position in {19, 22}:
            return k_position == 17
        elif q_position in {20}:
            return k_position == 4
        elif q_position in {21}:
            return k_position == 26
        elif q_position in {24}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 19
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {31}:
            return k_position == 37
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {33, 36, 38}:
            return k_position == 24
        elif q_position in {35}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 30
        elif q_position in {39}:
            return k_position == 31

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 3, 4}:
            return k_position == 7
        elif q_position in {1, 34, 33}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 19
        elif q_position in {5}:
            return k_position == 12
        elif q_position in {38, 6}:
            return k_position == 25
        elif q_position in {
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
            return k_position == 2
        elif q_position in {30}:
            return k_position == 29
        elif q_position in {39, 31}:
            return k_position == 9
        elif q_position in {32}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 18
        elif q_position in {36, 37}:
            return k_position == 14

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, positions)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0, 19, 12}:
            return token == "2"
        elif position in {1, 13}:
            return token == "4"
        elif position in {2, 3, 4, 5, 6, 10}:
            return token == "5"
        elif position in {
            7,
            8,
            9,
            11,
            14,
            15,
            17,
            18,
            21,
            22,
            23,
            24,
            25,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {16, 20, 26, 28, 29}:
            return token == "1"
        elif position in {27}:
            return token == "3"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 7, 13, 16, 22}:
            return token == "3"
        elif position in {1, 3, 4, 5, 6, 25, 26}:
            return token == "1"
        elif position in {2, 11, 20, 15}:
            return token == "4"
        elif position in {8, 9}:
            return token == "5"
        elif position in {
            10,
            12,
            14,
            17,
            18,
            19,
            21,
            23,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {24}:
            return token == "2"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {
            0,
            2,
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
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {1}:
            return token == "3"
        elif position in {5}:
            return token == "<s>"
        elif position in {6}:
            return token == "5"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1}:
            return token == "5"
        elif position in {2, 3, 4, 5, 6}:
            return token == "<s>"
        elif position in {
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
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2, 3, 4, 5, 32, 33, 34, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {6}:
            return token == "2"
        elif position in {21, 7}:
            return token == "5"
        elif position in {8, 9, 11, 12, 14, 15, 20, 24, 27, 29}:
            return token == "1"
        elif position in {10, 13, 16, 17, 18, 19, 22, 23, 25, 26, 28}:
            return token == "0"
        elif position in {35}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 2, 7}:
            return token == "4"
        elif position in {3, 4, 5}:
            return token == "<s>"
        elif position in {
            6,
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
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {
            0,
            2,
            3,
            4,
            5,
            21,
            22,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {
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
            23,
            24,
        }:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 32, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {1}:
            return token == "2"
        elif position in {7, 8, 11, 12, 15, 21, 22, 27}:
            return token == "5"
        elif position in {9, 14, 20, 24, 29}:
            return token == "1"
        elif position in {10, 13, 16, 17, 18, 23, 25, 26, 28}:
            return token == "0"
        elif position in {19}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 32, 2, 3, 33, 34, 6, 36, 30}:
            return token == ""
        elif position in {1, 18, 12}:
            return token == "3"
        elif position in {35, 4, 37, 5, 38, 39, 31}:
            return token == "<pad>"
        elif position in {7, 9, 11, 16, 17, 21, 22, 26}:
            return token == "2"
        elif position in {8, 10, 13, 14, 15, 19, 20, 23, 24}:
            return token == "5"
        elif position in {25, 28, 29}:
            return token == "4"
        elif position in {27}:
            return token == "0"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 5, 6}:
            return token == "1"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
            3,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {17, 4, 23}:
            return token == "<pad>"
        elif position in {8, 7}:
            return token == "<s>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_7_output):
        key = (attn_0_2_output, attn_0_7_output)
        if key in {
            (0, "1"),
            (2, "1"),
            (3, "1"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (5, "1"),
            (6, "1"),
            (7, "1"),
            (8, "0"),
            (8, "1"),
            (8, "2"),
            (8, "3"),
            (8, "4"),
            (8, "5"),
            (9, "1"),
            (10, "1"),
            (12, "1"),
            (13, "1"),
            (14, "1"),
            (16, "1"),
            (17, "1"),
            (17, "5"),
            (18, "0"),
            (18, "1"),
            (18, "2"),
            (18, "3"),
            (18, "4"),
            (18, "5"),
            (19, "0"),
            (19, "1"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "5"),
            (20, "1"),
            (21, "1"),
            (22, "0"),
            (22, "1"),
            (22, "2"),
            (22, "3"),
            (22, "4"),
            (22, "5"),
            (23, "0"),
            (23, "1"),
            (23, "2"),
            (23, "3"),
            (23, "4"),
            (23, "5"),
            (24, "1"),
            (25, "1"),
            (27, "1"),
            (28, "1"),
            (29, "1"),
            (30, "1"),
            (31, "1"),
            (32, "1"),
            (33, "1"),
            (34, "1"),
            (35, "1"),
            (36, "1"),
            (37, "1"),
            (38, "1"),
            (39, "1"),
        }:
            return 13
        return 6

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_7_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_4_output, attn_0_7_output):
        key = (attn_0_4_output, attn_0_7_output)
        if key in {("0", "0"), ("0", "4"), ("0", "<s>")}:
            return 0
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_7_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_2_output, attn_0_5_output):
        key = (attn_0_2_output, attn_0_5_output)
        return 7

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_5_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_6_output, token):
        key = (attn_0_6_output, token)
        if key in {
            ("0", "5"),
            ("1", "5"),
            ("2", "5"),
            ("3", "5"),
            ("4", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "<s>"),
            ("<s>", "5"),
        }:
            return 10
        return 5

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_6_outputs, tokens)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_4_output):
        key = num_attn_0_4_output
        return 7

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_4_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_4_output, num_attn_0_5_output):
        key = (num_attn_0_4_output, num_attn_0_5_output)
        return 39

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_1_output, num_attn_0_4_output):
        key = (num_attn_0_1_output, num_attn_0_4_output)
        return 31

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_4_output):
        key = (num_attn_0_2_output, num_attn_0_4_output)
        return 32

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "<s>"
        elif attn_0_4_output in {"1"}:
            return token == "4"
        elif attn_0_4_output in {"5", "<s>", "2", "4"}:
            return token == ""
        elif attn_0_4_output in {"3"}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, attn_0_4_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "<s>"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, attn_0_6_output):
        if token in {"0", "3", "1", "5"}:
            return attn_0_6_output == "2"
        elif token in {"<s>", "2"}:
            return attn_0_6_output == "5"
        elif token in {"4"}:
            return attn_0_6_output == ""

    attn_1_2_pattern = select_closest(attn_0_6_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, attn_0_3_output):
        if position in {0, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 23, 24}:
            return attn_0_3_output == "3"
        elif position in {1, 30, 39}:
            return attn_0_3_output == "4"
        elif position in {2, 29}:
            return attn_0_3_output == "1"
        elif position in {32, 3, 37, 31}:
            return attn_0_3_output == ""
        elif position in {5}:
            return attn_0_3_output == "2"
        elif position in {33, 34, 35, 36, 6, 38, 16, 21, 22, 25, 26}:
            return attn_0_3_output == "<s>"
        elif position in {27}:
            return attn_0_3_output == "0"
        elif position in {28}:
            return attn_0_3_output == "5"

    attn_1_3_pattern = select_closest(attn_0_3_outputs, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, attn_0_7_output):
        if position in {0}:
            return attn_0_7_output == "1"
        elif position in {1, 12, 23}:
            return attn_0_7_output == "5"
        elif position in {24, 2, 22}:
            return attn_0_7_output == "<s>"
        elif position in {3}:
            return attn_0_7_output == "4"
        elif position in {4}:
            return attn_0_7_output == "0"
        elif position in {
            5,
            7,
            8,
            9,
            10,
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
            25,
            27,
            29,
        }:
            return attn_0_7_output == "3"
        elif position in {32, 33, 34, 36, 6, 26, 28, 31}:
            return attn_0_7_output == ""
        elif position in {35, 37, 38, 39, 30}:
            return attn_0_7_output == "2"

    attn_1_4_pattern = select_closest(attn_0_7_outputs, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_4_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 3, 8, 9, 14, 16, 17, 18, 19, 21, 22, 23, 24, 29}:
            return token == "4"
        elif position in {4, 5, 6, 10, 13, 15}:
            return token == "3"
        elif position in {7, 11, 20, 26, 27, 28}:
            return token == "2"
        elif position in {25, 12}:
            return token == "<s>"
        elif position in {33, 34, 35, 37, 39, 30, 31}:
            return token == ""
        elif position in {32, 36, 38}:
            return token == "5"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"<s>", "1", "2", "5", "3", "4"}:
            return k_token == "0"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 3, 4}:
            return token == "4"
        elif position in {2}:
            return token == "1"
        elif position in {5, 6, 11, 23, 26}:
            return token == "3"
        elif position in {
            7,
            8,
            9,
            10,
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
            24,
            25,
            27,
            28,
            29,
            30,
            34,
            36,
            39,
        }:
            return token == "5"
        elif position in {32, 35, 37, 38, 31}:
            return token == ""
        elif position in {33}:
            return token == "<s>"

    attn_1_7_pattern = select_closest(tokens, positions, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 1, 32, 3, 4, 34, 35, 36, 37, 38, 39, 29, 30, 31}:
            return token == ""
        elif position in {33, 2}:
            return token == "<pad>"
        elif position in {5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 23, 24, 26, 28}:
            return token == "5"
        elif position in {7}:
            return token == "4"
        elif position in {10, 17, 21, 22, 27}:
            return token == "3"
        elif position in {25, 20}:
            return token == "2"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_4_output, attn_0_3_output):
        if attn_0_4_output in {"0", "2", "5", "3", "4"}:
            return attn_0_3_output == ""
        elif attn_0_4_output in {"1", "<s>"}:
            return attn_0_3_output == "1"

    num_attn_1_1_pattern = select(attn_0_3_outputs, attn_0_4_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(position, token):
        if position in {
            0,
            4,
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
            25,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {1}:
            return token == "1"
        elif position in {24, 26, 2, 3}:
            return token == "<pad>"

    num_attn_1_2_pattern = select(tokens, positions, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {
            0,
            2,
            4,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            28,
            30,
            31,
            32,
            33,
            35,
            36,
            38,
            39,
        }:
            return token == ""
        elif position in {1, 10, 12, 15, 16, 17, 27, 29}:
            return token == "5"
        elif position in {3}:
            return token == "<pad>"
        elif position in {19, 5, 6, 7}:
            return token == "0"
        elif position in {34, 37, 8, 11, 13, 14, 18}:
            return token == "4"
        elif position in {9}:
            return token == "<s>"

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(token, position):
        if token in {"0", "3"}:
            return position == 7
        elif token in {"1"}:
            return position == 28
        elif token in {"2"}:
            return position == 26
        elif token in {"4"}:
            return position == 39
        elif token in {"<s>", "5"}:
            return position == 33

    num_attn_1_4_pattern = select(positions, tokens, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_4_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(token, position):
        if token in {"0"}:
            return position == 21
        elif token in {"1"}:
            return position == 27
        elif token in {"<s>", "2", "5"}:
            return position == 7
        elif token in {"3"}:
            return position == 24
        elif token in {"4"}:
            return position == 10

    num_attn_1_5_pattern = select(positions, tokens, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(q_attn_0_7_output, k_attn_0_7_output):
        if q_attn_0_7_output in {"0", "<s>", "2", "5", "3", "4"}:
            return k_attn_0_7_output == "1"
        elif q_attn_0_7_output in {"1"}:
            return k_attn_0_7_output == "4"

    num_attn_1_6_pattern = select(attn_0_7_outputs, attn_0_7_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_1_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(token, attn_0_0_output):
        if token in {"0"}:
            return attn_0_0_output == "<s>"
        elif token in {"<s>", "1", "2", "5", "3", "4"}:
            return attn_0_0_output == ""

    num_attn_1_7_pattern = select(attn_0_0_outputs, tokens, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_4_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_6_output):
        key = (attn_1_2_output, attn_1_6_output)
        return 22

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_6_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, position):
        key = (attn_1_3_output, position)
        if key in {
            ("0", 4),
            ("0", 7),
            ("0", 8),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("0", 15),
            ("0", 16),
            ("0", 17),
            ("0", 18),
            ("0", 19),
            ("0", 23),
            ("0", 26),
            ("1", 4),
            ("1", 16),
            ("1", 23),
            ("2", 4),
            ("2", 7),
            ("2", 11),
            ("2", 16),
            ("2", 23),
            ("3", 0),
            ("3", 1),
            ("3", 2),
            ("3", 3),
            ("3", 4),
            ("3", 5),
            ("3", 6),
            ("3", 7),
            ("3", 8),
            ("3", 9),
            ("3", 10),
            ("3", 11),
            ("3", 12),
            ("3", 13),
            ("3", 14),
            ("3", 15),
            ("3", 16),
            ("3", 17),
            ("3", 18),
            ("3", 19),
            ("3", 20),
            ("3", 21),
            ("3", 22),
            ("3", 23),
            ("3", 24),
            ("3", 25),
            ("3", 26),
            ("3", 27),
            ("3", 28),
            ("3", 29),
            ("3", 30),
            ("3", 31),
            ("3", 32),
            ("3", 33),
            ("3", 34),
            ("3", 35),
            ("3", 36),
            ("3", 37),
            ("3", 38),
            ("3", 39),
            ("4", 4),
            ("4", 7),
            ("4", 11),
            ("4", 12),
            ("4", 15),
            ("4", 16),
            ("4", 17),
            ("4", 18),
            ("4", 19),
            ("4", 23),
            ("4", 26),
            ("5", 4),
            ("5", 7),
            ("5", 11),
            ("5", 16),
            ("5", 23),
            ("<s>", 4),
            ("<s>", 7),
            ("<s>", 11),
            ("<s>", 16),
            ("<s>", 23),
        }:
            return 28
        elif key in {
            ("0", 22),
            ("0", 29),
            ("1", 14),
            ("1", 22),
            ("1", 29),
            ("1", 36),
            ("2", 14),
            ("2", 22),
            ("4", 22),
            ("4", 29),
            ("4", 36),
            ("5", 14),
            ("5", 22),
            ("5", 29),
            ("<s>", 14),
            ("<s>", 22),
            ("<s>", 29),
        }:
            return 23
        elif key in {
            ("0", 5),
            ("0", 6),
            ("0", 14),
            ("0", 20),
            ("1", 5),
            ("1", 6),
            ("1", 20),
            ("2", 5),
            ("2", 6),
            ("2", 20),
            ("4", 5),
            ("4", 6),
            ("4", 14),
            ("4", 20),
            ("5", 5),
            ("5", 6),
            ("5", 20),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 20),
        }:
            return 6
        return 3

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, positions)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(position, attn_1_0_output):
        key = (position, attn_1_0_output)
        if key in {(39, 27)}:
            return 11
        return 19

    mlp_1_2_outputs = [mlp_1_2(k0, k1) for k0, k1 in zip(positions, attn_1_0_outputs)]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, attn_1_6_output):
        key = (position, attn_1_6_output)
        return 15

    mlp_1_3_outputs = [mlp_1_3(k0, k1) for k0, k1 in zip(positions, attn_1_6_outputs)]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_1_output, num_attn_1_0_output):
        key = (num_attn_1_1_output, num_attn_1_0_output)
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_1_0_output):
        key = (num_attn_1_2_output, num_attn_1_0_output)
        return 13

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        return 35

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_6_output):
        key = (num_attn_1_7_output, num_attn_1_6_output)
        return 5

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_6_output, position):
        if attn_0_6_output in {"0", "2"}:
            return position == 0
        elif attn_0_6_output in {"1"}:
            return position == 7
        elif attn_0_6_output in {"3"}:
            return position == 8
        elif attn_0_6_output in {"4"}:
            return position == 9
        elif attn_0_6_output in {"5"}:
            return position == 10
        elif attn_0_6_output in {"<s>"}:
            return position == 3

    attn_2_0_pattern = select_closest(positions, attn_0_6_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_3_output, mlp_1_1_output):
        if attn_1_3_output in {"0"}:
            return mlp_1_1_output == 23
        elif attn_1_3_output in {"1"}:
            return mlp_1_1_output == 22
        elif attn_1_3_output in {"2"}:
            return mlp_1_1_output == 16
        elif attn_1_3_output in {"3"}:
            return mlp_1_1_output == 38
        elif attn_1_3_output in {"5", "<s>", "4"}:
            return mlp_1_1_output == 21

    attn_2_1_pattern = select_closest(mlp_1_1_outputs, attn_1_3_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_4_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_1_output, num_mlp_1_0_output):
        if attn_0_1_output in {"0", "2"}:
            return num_mlp_1_0_output == 39
        elif attn_0_1_output in {"1"}:
            return num_mlp_1_0_output == 32
        elif attn_0_1_output in {"3"}:
            return num_mlp_1_0_output == 4
        elif attn_0_1_output in {"4"}:
            return num_mlp_1_0_output == 27
        elif attn_0_1_output in {"5"}:
            return num_mlp_1_0_output == 7
        elif attn_0_1_output in {"<s>"}:
            return num_mlp_1_0_output == 6

    attn_2_2_pattern = select_closest(
        num_mlp_1_0_outputs, attn_0_1_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(num_mlp_1_0_output, attn_0_6_output):
        if num_mlp_1_0_output in {0, 39, 24, 25, 30, 31}:
            return attn_0_6_output == "3"
        elif num_mlp_1_0_output in {1, 37, 6, 7, 26, 27}:
            return attn_0_6_output == "5"
        elif num_mlp_1_0_output in {
            2,
            3,
            4,
            5,
            8,
            9,
            10,
            11,
            12,
            15,
            16,
            17,
            18,
            20,
            21,
            23,
            28,
            29,
            32,
            34,
            35,
            36,
            38,
        }:
            return attn_0_6_output == ""
        elif num_mlp_1_0_output in {33, 19, 13}:
            return attn_0_6_output == "2"
        elif num_mlp_1_0_output in {14}:
            return attn_0_6_output == "<s>"
        elif num_mlp_1_0_output in {22}:
            return attn_0_6_output == "0"

    attn_2_3_pattern = select_closest(
        attn_0_6_outputs, num_mlp_1_0_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_4_output, token):
        if attn_0_4_output in {"0"}:
            return token == "1"
        elif attn_0_4_output in {"1"}:
            return token == "0"
        elif attn_0_4_output in {"3", "<s>", "2", "4"}:
            return token == ""
        elif attn_0_4_output in {"5"}:
            return token == "4"

    attn_2_4_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, tokens)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_1_3_output, token):
        if attn_1_3_output in {"0"}:
            return token == "1"
        elif attn_1_3_output in {"1"}:
            return token == "4"
        elif attn_1_3_output in {"<s>", "2"}:
            return token == ""
        elif attn_1_3_output in {"3"}:
            return token == "3"
        elif attn_1_3_output in {"4"}:
            return token == "0"
        elif attn_1_3_output in {"5"}:
            return token == "5"

    attn_2_5_pattern = select_closest(tokens, attn_1_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 3
        elif attn_0_2_output in {8, 1, 6}:
            return position == 9
        elif attn_0_2_output in {16, 2, 18}:
            return position == 8
        elif attn_0_2_output in {3, 38}:
            return position == 12
        elif attn_0_2_output in {4}:
            return position == 19
        elif attn_0_2_output in {33, 5, 15}:
            return position == 2
        elif attn_0_2_output in {24, 7}:
            return position == 7
        elif attn_0_2_output in {9, 27}:
            return position == 10
        elif attn_0_2_output in {10, 13}:
            return position == 15
        elif attn_0_2_output in {11, 28}:
            return position == 28
        elif attn_0_2_output in {12}:
            return position == 13
        elif attn_0_2_output in {14}:
            return position == 11
        elif attn_0_2_output in {17, 19}:
            return position == 4
        elif attn_0_2_output in {36, 37, 39, 20, 23, 30, 31}:
            return position == 1
        elif attn_0_2_output in {21}:
            return position == 26
        elif attn_0_2_output in {26, 34, 22}:
            return position == 0
        elif attn_0_2_output in {25, 29}:
            return position == 14
        elif attn_0_2_output in {32}:
            return position == 21
        elif attn_0_2_output in {35}:
            return position == 20

    attn_2_6_pattern = select_closest(positions, attn_0_2_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_3_output, token):
        if attn_0_3_output in {"0", "2"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "4"
        elif attn_0_3_output in {"3", "<s>", "5"}:
            return token == ""
        elif attn_0_3_output in {"4"}:
            return token == "2"

    attn_2_7_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {"0", "<s>", "1", "2", "5", "3"}:
            return attn_0_1_output == ""
        elif attn_0_3_output in {"4"}:
            return attn_0_1_output == "4"

    num_attn_2_0_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_1_4_output):
        if attn_1_1_output in {"0", "<s>", "1", "2", "5", "3"}:
            return attn_1_4_output == ""
        elif attn_1_1_output in {"4"}:
            return attn_1_4_output == "4"

    num_attn_2_1_pattern = select(attn_1_4_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(token, position):
        if token in {"0"}:
            return position == 33
        elif token in {"1", "5"}:
            return position == 7
        elif token in {"2"}:
            return position == 39
        elif token in {"3"}:
            return position == 32
        elif token in {"4"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 38

    num_attn_2_2_pattern = select(positions, tokens, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_4_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(token, attn_0_5_output):
        if token in {"0"}:
            return attn_0_5_output == 10
        elif token in {"1"}:
            return attn_0_5_output == 36
        elif token in {"2"}:
            return attn_0_5_output == 31
        elif token in {"3"}:
            return attn_0_5_output == 22
        elif token in {"4"}:
            return attn_0_5_output == 9
        elif token in {"5"}:
            return attn_0_5_output == 8
        elif token in {"<s>"}:
            return attn_0_5_output == 25

    num_attn_2_3_pattern = select(attn_0_5_outputs, tokens, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(position, attn_1_1_output):
        if position in {0, 1, 2, 3, 4, 5, 32, 33, 34, 35, 36, 37, 38, 30}:
            return attn_1_1_output == ""
        elif position in {
            6,
            7,
            39,
            12,
            13,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            26,
            28,
            29,
            31,
        }:
            return attn_1_1_output == "4"
        elif position in {8, 9, 10, 11, 14, 15, 21, 25, 27}:
            return attn_1_1_output == "0"

    num_attn_2_4_pattern = select(attn_1_1_outputs, positions, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_3_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, attn_1_4_output):
        if position in {0, 32, 2, 3, 4, 5, 6, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return attn_1_4_output == ""
        elif position in {1, 7, 9, 12, 13, 19, 20, 21, 24}:
            return attn_1_4_output == "4"
        elif position in {8, 11, 14, 15, 27, 29}:
            return attn_1_4_output == "0"
        elif position in {10, 16, 18, 22, 23, 25, 26, 28}:
            return attn_1_4_output == "1"
        elif position in {17}:
            return attn_1_4_output == "5"

    num_attn_2_5_pattern = select(attn_1_4_outputs, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_3_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_1_output, attn_0_3_output):
        if attn_1_1_output in {"0"}:
            return attn_0_3_output == "1"
        elif attn_1_1_output in {"3", "1", "2", "5"}:
            return attn_0_3_output == "0"
        elif attn_1_1_output in {"<s>", "4"}:
            return attn_0_3_output == ""

    num_attn_2_6_pattern = select(attn_0_3_outputs, attn_1_1_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_3_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_6_output, attn_0_0_output):
        if attn_0_6_output in {"0", "<s>", "1", "2", "3", "4"}:
            return attn_0_0_output == ""
        elif attn_0_6_output in {"5"}:
            return attn_0_0_output == "5"

    num_attn_2_7_pattern = select(attn_0_0_outputs, attn_0_6_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_1_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_0_output):
        key = mlp_0_0_output
        if key in {10}:
            return 21
        return 23

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in mlp_0_0_outputs]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, attn_1_2_output):
        key = (num_mlp_0_1_output, attn_1_2_output)
        return 15

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_1_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_1_6_output, attn_2_4_output):
        key = (attn_1_6_output, attn_2_4_output)
        return 5

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_1_6_outputs, attn_2_4_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_0_7_output, mlp_1_0_output):
        key = (attn_0_7_output, mlp_1_0_output)
        if key in {("4", 19)}:
            return 35
        return 29

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_0_7_outputs, mlp_1_0_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_6_output, num_attn_1_7_output):
        key = (num_attn_1_6_output, num_attn_1_7_output)
        return 13

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output, num_attn_2_6_output):
        key = (num_attn_2_1_output, num_attn_2_6_output)
        return 36

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_1_outputs, num_attn_2_6_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_4_output, num_attn_2_7_output):
        key = (num_attn_2_4_output, num_attn_2_7_output)
        return 27

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_4_outputs, num_attn_2_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_3_output, num_attn_2_4_output):
        key = (num_attn_1_3_output, num_attn_2_4_output)
        return 3

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_4_outputs)
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


print(run(["<s>", "5", "0", "3", "2", "3", "0", "2", "1", "3"]))

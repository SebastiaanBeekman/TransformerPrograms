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
        "output/length/rasp/mostfreq/trainlength30/s3/most_freq_weights.csv",
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
        if position in {0, 7, 8, 9, 10, 13, 15, 16, 17, 18, 20, 22, 24, 27, 28, 29}:
            return token == "4"
        elif position in {1, 26, 19}:
            return token == "1"
        elif position in {2, 3, 4}:
            return token == "3"
        elif position in {5, 6}:
            return token == "0"
        elif position in {11, 12, 14}:
            return token == "5"
        elif position in {25, 21}:
            return token == "<s>"
        elif position in {32, 33, 34, 35, 36, 37, 23, 30, 31}:
            return token == ""
        elif position in {38, 39}:
            return token == "2"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 8}:
            return token == "3"
        elif position in {1, 15, 17, 26, 28, 29}:
            return token == "4"
        elif position in {2, 3, 4, 5, 6}:
            return token == "0"
        elif position in {7, 10, 11, 13, 16, 18, 19, 22, 23, 25}:
            return token == "2"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 9, 12, 14, 24, 27, 30, 31}:
            return token == ""
        elif position in {20}:
            return token == "<pad>"
        elif position in {21}:
            return token == "<s>"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {
            0,
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
            return token == "4"
        elif position in {1}:
            return token == "5"
        elif position in {2, 3, 4, 5, 6}:
            return token == "3"
        elif position in {32, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {33, 34}:
            return token == "<pad>"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1, 7, 9, 10, 13, 18, 20, 22, 24, 25, 27}:
            return token == "4"
        elif position in {8, 2, 28}:
            return token == "2"
        elif position in {3, 4, 5, 11, 12, 14, 15, 16, 17, 21, 29}:
            return token == "5"
        elif position in {6}:
            return token == "0"
        elif position in {19}:
            return token == "<s>"
        elif position in {32, 33, 34, 35, 36, 37, 38, 39, 23, 26, 30, 31}:
            return token == ""

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 17}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3, 28}:
            return k_position == 2
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {6, 15}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {8}:
            return k_position == 28
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {24, 29, 11, 13}:
            return k_position == 14
        elif q_position in {16, 12}:
            return k_position == 18
        elif q_position in {38, 14}:
            return k_position == 24
        elif q_position in {18, 20}:
            return k_position == 22
        elif q_position in {19, 31, 23}:
            return k_position == 17
        elif q_position in {25, 21}:
            return k_position == 0
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 11
        elif q_position in {27}:
            return k_position == 12
        elif q_position in {30}:
            return k_position == 23
        elif q_position in {32}:
            return k_position == 38
        elif q_position in {33}:
            return k_position == 30
        elif q_position in {34, 37}:
            return k_position == 27
        elif q_position in {35}:
            return k_position == 39
        elif q_position in {36}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 34

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 6
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 10}:
            return k_position == 18
        elif q_position in {32, 4, 39}:
            return k_position == 9
        elif q_position in {5, 6}:
            return k_position == 23
        elif q_position in {
            7,
            8,
            9,
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
            return k_position == 17
        elif q_position in {35, 36, 30, 31}:
            return k_position == 8
        elif q_position in {33}:
            return k_position == 38
        elif q_position in {34}:
            return k_position == 28
        elif q_position in {37}:
            return k_position == 31
        elif q_position in {38}:
            return k_position == 12

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, positions)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "3"
        elif position in {2, 23}:
            return token == "1"
        elif position in {3, 4, 6, 11, 13, 14, 15, 17, 20, 26, 27}:
            return token == "5"
        elif position in {5, 7, 8, 9, 12, 16, 18, 22, 24, 28, 29}:
            return token == "4"
        elif position in {32, 33, 34, 35, 36, 37, 38, 10, 19, 21, 25, 30, 31}:
            return token == ""
        elif position in {39}:
            return token == "0"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {8, 3}:
            return k_position == 4
        elif q_position in {4, 7}:
            return k_position == 5
        elif q_position in {9, 5}:
            return k_position == 6
        elif q_position in {24, 11, 6}:
            return k_position == 12
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 23
        elif q_position in {18, 13}:
            return k_position == 8
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 11
        elif q_position in {36, 39, 16, 20, 29}:
            return k_position == 17
        elif q_position in {17, 35}:
            return k_position == 21
        elif q_position in {19, 21, 23, 25, 26, 27}:
            return k_position == 0
        elif q_position in {22}:
            return k_position == 22
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {30}:
            return k_position == 36
        elif q_position in {33, 31}:
            return k_position == 39
        elif q_position in {32}:
            return k_position == 33
        elif q_position in {34}:
            return k_position == 29
        elif q_position in {37}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 25

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 6, 13, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29}:
            return token == "<s>"
        elif position in {1}:
            return token == "5"
        elif position in {33, 2, 3, 4, 5, 34, 35, 36, 37, 38, 39, 22, 24, 30, 31}:
            return token == ""
        elif position in {10, 15, 14, 7}:
            return token == "2"
        elif position in {8, 9, 11}:
            return token == "1"
        elif position in {12}:
            return token == "4"
        elif position in {32}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1, 7, 8, 9, 10}:
            return token == "2"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            14,
            16,
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
        elif position in {11, 12, 13}:
            return token == "<s>"
        elif position in {22, 15}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 1, 2}:
            return token == "4"
        elif position in {3, 4, 5, 6}:
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
        elif position in {17, 18}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 1, 7, 8, 9}:
            return token == "1"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            13,
            16,
            17,
            18,
            19,
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
        elif position in {10, 11, 14, 15, 20}:
            return token == "<s>"
        elif position in {12, 31}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
            12,
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
        elif position in {5, 6, 7, 8, 9, 10, 11, 13, 14, 15}:
            return token == "<s>"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 17, 18, 27}:
            return token == "<s>"
        elif position in {1, 20, 22, 26, 28, 29}:
            return token == "3"
        elif position in {32, 33, 2, 3, 4, 5, 6, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {10, 12, 15, 7}:
            return token == "1"
        elif position in {8, 23}:
            return token == "2"
        elif position in {9, 21}:
            return token == "0"
        elif position in {11, 13, 14, 16, 19, 24, 25}:
            return token == "4"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1, 2, 3}:
            return token == "3"
        elif position in {
            4,
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
        elif position in {5, 7}:
            return token == "<s>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 2, 3, 4, 5, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "5"
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
        elif position in {20}:
            return token == "<pad>"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(position, attn_0_3_output):
        key = (position, attn_0_3_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "<s>"),
            (4, "2"),
            (4, "4"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "<s>"),
            (11, "2"),
            (13, "2"),
            (15, "2"),
            (15, "<s>"),
            (17, "2"),
            (18, "2"),
            (19, "1"),
            (19, "2"),
            (19, "3"),
            (19, "4"),
            (19, "<s>"),
            (20, "1"),
            (20, "2"),
            (20, "3"),
            (20, "4"),
            (20, "<s>"),
            (23, "2"),
            (24, "1"),
            (24, "2"),
            (24, "3"),
            (24, "4"),
            (24, "<s>"),
            (25, "1"),
            (25, "2"),
            (25, "3"),
            (25, "4"),
            (25, "<s>"),
            (26, "2"),
            (26, "4"),
            (28, "0"),
            (28, "1"),
            (28, "2"),
            (28, "4"),
            (28, "<s>"),
            (30, "2"),
            (30, "3"),
            (31, "2"),
            (32, "2"),
            (33, "2"),
            (34, "2"),
            (35, "2"),
            (36, "2"),
            (37, "2"),
            (37, "3"),
            (38, "2"),
            (39, "2"),
        }:
            return 29
        elif key in {
            (6, "0"),
            (6, "1"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "<s>"),
            (9, "<s>"),
            (12, "<s>"),
            (23, "<s>"),
            (26, "0"),
            (26, "1"),
            (26, "3"),
            (26, "<s>"),
            (29, "2"),
            (29, "<s>"),
            (30, "<s>"),
            (31, "<s>"),
            (32, "<s>"),
            (33, "<s>"),
            (34, "<s>"),
            (35, "<s>"),
            (36, "<s>"),
            (37, "<s>"),
            (38, "<s>"),
            (39, "<s>"),
        }:
            return 17
        elif key in {
            (18, "1"),
            (18, "<s>"),
            (22, "2"),
            (24, "0"),
            (25, "0"),
            (30, "1"),
            (31, "1"),
            (32, "1"),
            (33, "1"),
            (34, "1"),
            (36, "1"),
            (37, "1"),
            (38, "1"),
            (39, "1"),
        }:
            return 7
        elif key in {
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (2, "5"),
        }:
            return 32
        elif key in {(4, "1"), (4, "3"), (4, "<s>"), (28, "3")}:
            return 35
        elif key in {(2, "1"), (2, "2"), (2, "3")}:
            return 0
        elif key in {
            (0, "5"),
            (2, "4"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
        }:
            return 22
        elif key in {(2, "<s>")}:
            return 27
        return 38

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(positions, attn_0_3_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, attn_0_4_output):
        key = (attn_0_1_output, attn_0_4_output)
        if key in {
            ("1", "4"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "5"),
            ("4", "<s>"),
        }:
            return 16
        elif key in {
            ("4", "4"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 26
        return 13

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_4_output, attn_0_2_output):
        key = (attn_0_4_output, attn_0_2_output)
        return 14

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_2_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position, attn_0_2_output):
        key = (position, attn_0_2_output)
        if key in {
            (0, "0"),
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "4"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (5, "0"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (6, "0"),
            (6, "1"),
            (6, "2"),
            (6, "3"),
            (6, "4"),
            (6, "5"),
            (6, "<s>"),
            (7, "3"),
            (9, "3"),
            (10, "3"),
            (11, "3"),
            (13, "3"),
            (14, "3"),
            (15, "3"),
            (17, "3"),
            (18, "3"),
            (27, "3"),
            (30, "0"),
            (30, "1"),
            (30, "2"),
            (30, "3"),
            (30, "4"),
            (30, "5"),
            (31, "0"),
            (31, "1"),
            (31, "2"),
            (31, "3"),
            (31, "4"),
            (31, "5"),
            (32, "0"),
            (32, "1"),
            (32, "2"),
            (32, "3"),
            (32, "4"),
            (32, "5"),
            (32, "<s>"),
            (33, "0"),
            (33, "1"),
            (33, "2"),
            (33, "3"),
            (33, "4"),
            (33, "5"),
            (33, "<s>"),
            (34, "0"),
            (34, "1"),
            (34, "2"),
            (34, "3"),
            (34, "4"),
            (34, "5"),
            (34, "<s>"),
            (35, "0"),
            (35, "1"),
            (35, "2"),
            (35, "3"),
            (35, "4"),
            (35, "5"),
            (35, "<s>"),
            (36, "0"),
            (36, "1"),
            (36, "2"),
            (36, "3"),
            (36, "4"),
            (36, "5"),
            (36, "<s>"),
            (37, "0"),
            (37, "1"),
            (37, "2"),
            (37, "3"),
            (37, "4"),
            (37, "5"),
            (38, "0"),
            (38, "1"),
            (38, "2"),
            (38, "3"),
            (38, "4"),
            (38, "5"),
            (38, "<s>"),
            (39, "0"),
            (39, "1"),
            (39, "2"),
            (39, "3"),
            (39, "4"),
            (39, "5"),
            (39, "<s>"),
        }:
            return 3
        return 30

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(positions, attn_0_2_outputs)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_4_output):
        key = (num_attn_0_5_output, num_attn_0_4_output)
        return 37

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 11

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 39

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 6

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {"1", "0", "<s>", "5"}:
            return mlp_0_0_output == 27
        elif attn_0_1_output in {"2"}:
            return mlp_0_0_output == 10
        elif attn_0_1_output in {"3", "4"}:
            return mlp_0_0_output == 6

    attn_1_0_pattern = select_closest(mlp_0_0_outputs, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, mlp_0_0_output):
        if position in {0}:
            return mlp_0_0_output == 32
        elif position in {1, 12, 9}:
            return mlp_0_0_output == 10
        elif position in {2}:
            return mlp_0_0_output == 17
        elif position in {3, 29}:
            return mlp_0_0_output == 38
        elif position in {4, 7}:
            return mlp_0_0_output == 29
        elif position in {32, 5, 39, 10, 15, 17, 19, 20, 23, 24, 25, 26, 28, 30}:
            return mlp_0_0_output == 23
        elif position in {6}:
            return mlp_0_0_output == 39
        elif position in {8, 16}:
            return mlp_0_0_output == 7
        elif position in {33, 34, 11}:
            return mlp_0_0_output == 12
        elif position in {36, 13}:
            return mlp_0_0_output == 15
        elif position in {14}:
            return mlp_0_0_output == 28
        elif position in {18, 37, 22, 31}:
            return mlp_0_0_output == 27
        elif position in {21}:
            return mlp_0_0_output == 16
        elif position in {35, 27}:
            return mlp_0_0_output == 18
        elif position in {38}:
            return mlp_0_0_output == 13

    attn_1_1_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_4_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "4"
        elif q_token in {"5", "3", "2", "4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"<s>", "0", "2"}:
            return position == 0
        elif token in {"1"}:
            return position == 1
        elif token in {"3", "5"}:
            return position == 9
        elif token in {"4"}:
            return position == 4

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, mlp_0_0_output):
        if position in {0, 4, 14, 15, 22}:
            return mlp_0_0_output == 0
        elif position in {1, 19, 13, 17}:
            return mlp_0_0_output == 29
        elif position in {9, 2}:
            return mlp_0_0_output == 17
        elif position in {3, 37, 8, 20, 26}:
            return mlp_0_0_output == 32
        elif position in {21, 11, 5}:
            return mlp_0_0_output == 30
        elif position in {28, 6}:
            return mlp_0_0_output == 22
        elif position in {7}:
            return mlp_0_0_output == 10
        elif position in {16, 10, 18}:
            return mlp_0_0_output == 27
        elif position in {34, 12, 23}:
            return mlp_0_0_output == 25
        elif position in {24}:
            return mlp_0_0_output == 28
        elif position in {25}:
            return mlp_0_0_output == 23
        elif position in {27}:
            return mlp_0_0_output == 14
        elif position in {29}:
            return mlp_0_0_output == 20
        elif position in {30}:
            return mlp_0_0_output == 31
        elif position in {31}:
            return mlp_0_0_output == 37
        elif position in {32, 33}:
            return mlp_0_0_output == 35
        elif position in {35}:
            return mlp_0_0_output == 33
        elif position in {36}:
            return mlp_0_0_output == 15
        elif position in {38}:
            return mlp_0_0_output == 24
        elif position in {39}:
            return mlp_0_0_output == 26

    attn_1_4_pattern = select_closest(mlp_0_0_outputs, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_4_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(token, mlp_0_0_output):
        if token in {"0", "4", "2"}:
            return mlp_0_0_output == 17
        elif token in {"1"}:
            return mlp_0_0_output == 27
        elif token in {"3"}:
            return mlp_0_0_output == 10
        elif token in {"5"}:
            return mlp_0_0_output == 29
        elif token in {"<s>"}:
            return mlp_0_0_output == 13

    attn_1_5_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, positions)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_token, k_token):
        if q_token in {"0"}:
            return k_token == ""
        elif q_token in {"1", "3", "5"}:
            return k_token == "4"
        elif q_token in {"<s>", "2"}:
            return k_token == "<s>"
        elif q_token in {"4"}:
            return k_token == "5"

    attn_1_6_pattern = select_closest(tokens, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_6_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, mlp_0_0_output):
        if token in {"0", "3", "2"}:
            return mlp_0_0_output == 22
        elif token in {"1"}:
            return mlp_0_0_output == 10
        elif token in {"4"}:
            return mlp_0_0_output == 38
        elif token in {"5"}:
            return mlp_0_0_output == 32
        elif token in {"<s>"}:
            return mlp_0_0_output == 14

    attn_1_7_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, mlp_0_0_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_7_output):
        if position in {0, 1, 2, 32}:
            return attn_0_7_output == "1"
        elif position in {
            3,
            4,
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
            18,
            19,
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
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return attn_0_7_output == ""
        elif position in {5}:
            return attn_0_7_output == "<s>"
        elif position in {17, 20}:
            return attn_0_7_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_7_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
        if position in {0, 1, 2, 3, 4, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {10, 26, 5, 6}:
            return token == "5"
        elif position in {
            7,
            9,
            11,
            13,
            14,
            15,
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
        }:
            return token == "1"
        elif position in {8, 16, 12}:
            return token == "2"
        elif position in {23}:
            return token == "0"
        elif position in {32}:
            return token == "<pad>"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_7_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            11,
            12,
            14,
            16,
            19,
            20,
            22,
            24,
            25,
            26,
            27,
            28,
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
            return attn_0_0_output == ""
        elif mlp_0_0_output in {17, 10, 29, 5}:
            return attn_0_0_output == "4"
        elif mlp_0_0_output in {9, 15, 18, 21, 23}:
            return attn_0_0_output == "<pad>"
        elif mlp_0_0_output in {13}:
            return attn_0_0_output == "<s>"

    num_attn_1_2_pattern = select(attn_0_0_outputs, mlp_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(position, token):
        if position in {0}:
            return token == "2"
        elif position in {1}:
            return token == "3"
        elif position in {2, 3, 4}:
            return token == "<s>"
        elif position in {5, 6}:
            return token == "5"
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

    num_attn_1_3_pattern = select(tokens, positions, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {
            2,
            4,
            8,
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
            22,
            23,
            24,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
            38,
            39,
        }:
            return token == ""
        elif position in {16, 25, 26, 3}:
            return token == "<pad>"
        elif position in {35, 5, 6}:
            return token == "0"
        elif position in {7}:
            return token == "<s>"

    num_attn_1_4_pattern = select(tokens, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, ones)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_6_output, token):
        if attn_0_6_output in {"1", "0"}:
            return token == "<pad>"
        elif attn_0_6_output in {"<s>", "3", "2", "4"}:
            return token == ""
        elif attn_0_6_output in {"5"}:
            return token == "5"

    num_attn_1_5_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_7_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, attn_0_0_output):
        if position in {0}:
            return attn_0_0_output == "<s>"
        elif position in {32, 1, 7, 8, 12, 15}:
            return attn_0_0_output == "4"
        elif position in {
            2,
            3,
            4,
            5,
            6,
            18,
            19,
            21,
            23,
            24,
            26,
            27,
            28,
            30,
            31,
            34,
            35,
            36,
            39,
        }:
            return attn_0_0_output == ""
        elif position in {9, 14, 25}:
            return attn_0_0_output == "2"
        elif position in {10}:
            return attn_0_0_output == "3"
        elif position in {17, 11, 29}:
            return attn_0_0_output == "0"
        elif position in {13}:
            return attn_0_0_output == "5"
        elif position in {16, 20}:
            return attn_0_0_output == "1"
        elif position in {38, 33, 37, 22}:
            return attn_0_0_output == "<pad>"

    num_attn_1_6_pattern = select(attn_0_0_outputs, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_2_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_6_output, attn_0_7_output):
        if attn_0_6_output in {"<s>", "1", "3", "0", "2", "5"}:
            return attn_0_7_output == ""
        elif attn_0_6_output in {"4"}:
            return attn_0_7_output == "4"

    num_attn_1_7_pattern = select(attn_0_7_outputs, attn_0_6_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_3_output, attn_1_5_output):
        key = (mlp_0_3_output, attn_1_5_output)
        return 8

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, attn_1_5_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(token, attn_1_1_output):
        key = (token, attn_1_1_output)
        return 5

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(tokens, attn_1_1_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(num_mlp_0_3_output, mlp_0_2_output):
        key = (num_mlp_0_3_output, mlp_0_2_output)
        if key in {
            (5, 25),
            (9, 25),
            (10, 18),
            (10, 21),
            (10, 25),
            (10, 36),
            (15, 25),
            (17, 25),
            (18, 25),
            (21, 25),
            (23, 25),
            (25, 21),
            (25, 25),
            (28, 25),
            (29, 25),
            (36, 21),
            (36, 25),
            (36, 36),
            (37, 21),
            (37, 25),
        }:
            return 23
        return 37

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, mlp_0_2_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_4_output, attn_1_6_output):
        key = (attn_1_4_output, attn_1_6_output)
        return 2

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_1_6_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_7_output):
        key = (num_attn_1_4_output, num_attn_1_7_output)
        return 0

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_0_1_output):
        key = (num_attn_1_0_output, num_attn_0_1_output)
        return 32

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_4_output, num_attn_1_7_output):
        key = (num_attn_1_4_output, num_attn_1_7_output)
        return 29

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_3_output, num_attn_1_0_output):
        key = (num_attn_0_3_output, num_attn_1_0_output)
        return 38

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_6_output, token):
        if attn_0_6_output in {"0", "5"}:
            return token == "0"
        elif attn_0_6_output in {"1"}:
            return token == "3"
        elif attn_0_6_output in {"<s>", "2"}:
            return token == ""
        elif attn_0_6_output in {"3"}:
            return token == "4"
        elif attn_0_6_output in {"4"}:
            return token == "5"

    attn_2_0_pattern = select_closest(tokens, attn_0_6_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"0", "3", "2", "4"}:
            return token == "1"
        elif attn_0_1_output in {"1"}:
            return token == "4"
        elif attn_0_1_output in {"<s>", "5"}:
            return token == ""

    attn_2_1_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, tokens)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "<pad>"
        elif q_token in {"1", "3", "<s>", "5"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "5"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_1_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == ""
        elif attn_0_2_output in {"1", "<s>"}:
            return token == "4"
        elif attn_0_2_output in {"3", "2", "5"}:
            return token == "1"
        elif attn_0_2_output in {"4"}:
            return token == "5"

    attn_2_3_pattern = select_closest(tokens, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_5_output, token):
        if attn_0_5_output in {0, 17, 25, 39}:
            return token == "<s>"
        elif attn_0_5_output in {1, 37, 9}:
            return token == "2"
        elif attn_0_5_output in {2, 34, 7, 12, 27}:
            return token == "5"
        elif attn_0_5_output in {
            32,
            33,
            3,
            4,
            35,
            36,
            8,
            10,
            11,
            13,
            15,
            18,
            19,
            21,
            22,
        }:
            return token == ""
        elif attn_0_5_output in {5, 23}:
            return token == "3"
        elif attn_0_5_output in {6, 38, 20, 24, 28, 30}:
            return token == "4"
        elif attn_0_5_output in {14}:
            return token == "1"
        elif attn_0_5_output in {16, 26, 29, 31}:
            return token == "0"

    attn_2_4_pattern = select_closest(tokens, attn_0_5_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_4_output, mlp_0_0_output):
        if attn_0_4_output in {"<s>", "1", "3", "0", "5"}:
            return mlp_0_0_output == 27
        elif attn_0_4_output in {"2"}:
            return mlp_0_0_output == 12
        elif attn_0_4_output in {"4"}:
            return mlp_0_0_output == 29

    attn_2_5_pattern = select_closest(mlp_0_0_outputs, attn_0_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_4_output, token):
        if attn_0_4_output in {"0", "3", "<s>", "5"}:
            return token == ""
        elif attn_0_4_output in {"1"}:
            return token == "4"
        elif attn_0_4_output in {"2"}:
            return token == "3"
        elif attn_0_4_output in {"4"}:
            return token == "1"

    attn_2_6_pattern = select_closest(tokens, attn_0_4_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, tokens)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {0}:
            return k_mlp_0_0_output == 10
        elif q_mlp_0_0_output in {1, 2, 4, 5, 7}:
            return k_mlp_0_0_output == 35
        elif q_mlp_0_0_output in {3}:
            return k_mlp_0_0_output == 17
        elif q_mlp_0_0_output in {6}:
            return k_mlp_0_0_output == 23
        elif q_mlp_0_0_output in {8, 39, 31}:
            return k_mlp_0_0_output == 19
        elif q_mlp_0_0_output in {9, 11, 16, 18, 19, 26, 29}:
            return k_mlp_0_0_output == 27
        elif q_mlp_0_0_output in {34, 35, 36, 10, 21, 22, 27, 30}:
            return k_mlp_0_0_output == 32
        elif q_mlp_0_0_output in {12}:
            return k_mlp_0_0_output == 29
        elif q_mlp_0_0_output in {13}:
            return k_mlp_0_0_output == 25
        elif q_mlp_0_0_output in {28, 14}:
            return k_mlp_0_0_output == 12
        elif q_mlp_0_0_output in {37, 15}:
            return k_mlp_0_0_output == 28
        elif q_mlp_0_0_output in {17}:
            return k_mlp_0_0_output == 38
        elif q_mlp_0_0_output in {20}:
            return k_mlp_0_0_output == 14
        elif q_mlp_0_0_output in {23}:
            return k_mlp_0_0_output == 36
        elif q_mlp_0_0_output in {24}:
            return k_mlp_0_0_output == 0
        elif q_mlp_0_0_output in {25, 33}:
            return k_mlp_0_0_output == 18
        elif q_mlp_0_0_output in {32}:
            return k_mlp_0_0_output == 22
        elif q_mlp_0_0_output in {38}:
            return k_mlp_0_0_output == 31

    attn_2_7_pattern = select_closest(mlp_0_0_outputs, mlp_0_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_5_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_7_output, attn_0_2_output):
        if attn_0_7_output in {"<s>", "1", "3", "0", "2", "5"}:
            return attn_0_2_output == ""
        elif attn_0_7_output in {"4"}:
            return attn_0_2_output == "4"

    num_attn_2_0_pattern = select(attn_0_2_outputs, attn_0_7_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_2_output, attn_0_7_output):
        if attn_0_2_output in {"0"}:
            return attn_0_7_output == "<pad>"
        elif attn_0_2_output in {"4", "<s>", "1", "2", "5"}:
            return attn_0_7_output == ""
        elif attn_0_2_output in {"3"}:
            return attn_0_7_output == "3"

    num_attn_2_1_pattern = select(attn_0_7_outputs, attn_0_2_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(token, position):
        if token in {"0"}:
            return position == 5
        elif token in {"1"}:
            return position == 37
        elif token in {"2"}:
            return position == 22
        elif token in {"3"}:
            return position == 33
        elif token in {"4"}:
            return position == 32
        elif token in {"5"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 8

    num_attn_2_2_pattern = select(positions, tokens, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_4_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_4_output, attn_0_6_output):
        if attn_0_4_output in {"1", "0", "4", "5"}:
            return attn_0_6_output == ""
        elif attn_0_4_output in {"2"}:
            return attn_0_6_output == "2"
        elif attn_0_4_output in {"3", "<s>"}:
            return attn_0_6_output == "<s>"

    num_attn_2_3_pattern = select(attn_0_6_outputs, attn_0_4_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(mlp_0_0_output, attn_0_1_output):
        if mlp_0_0_output in {0, 16, 13, 38}:
            return attn_0_1_output == "<pad>"
        elif mlp_0_0_output in {
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
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
            30,
            31,
            32,
            34,
            36,
            37,
            39,
        }:
            return attn_0_1_output == ""
        elif mlp_0_0_output in {33, 35, 4, 10, 17, 18, 29}:
            return attn_0_1_output == "3"

    num_attn_2_4_pattern = select(attn_0_1_outputs, mlp_0_0_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_6_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(token, position):
        if token in {"0"}:
            return position == 6
        elif token in {"1", "5"}:
            return position == 37
        elif token in {"2"}:
            return position == 33
        elif token in {"3"}:
            return position == 21
        elif token in {"4"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 31

    num_attn_2_5_pattern = select(positions, tokens, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_4_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(mlp_0_0_output, attn_0_7_output):
        if mlp_0_0_output in {
            0,
            1,
            2,
            4,
            5,
            6,
            7,
            9,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            22,
            26,
            28,
            30,
            31,
            32,
            33,
            36,
            38,
            39,
        }:
            return attn_0_7_output == ""
        elif mlp_0_0_output in {34, 3, 35, 37, 10, 11, 17, 20, 21, 23, 24, 29}:
            return attn_0_7_output == "2"
        elif mlp_0_0_output in {8}:
            return attn_0_7_output == "1"
        elif mlp_0_0_output in {25}:
            return attn_0_7_output == "<pad>"
        elif mlp_0_0_output in {27}:
            return attn_0_7_output == "4"

    num_attn_2_6_pattern = select(attn_0_7_outputs, mlp_0_0_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(mlp_0_0_output, attn_0_0_output):
        if mlp_0_0_output in {
            0,
            1,
            3,
            4,
            5,
            6,
            7,
            8,
            11,
            12,
            15,
            16,
            19,
            20,
            22,
            25,
            27,
            31,
            32,
            34,
            39,
        }:
            return attn_0_0_output == ""
        elif mlp_0_0_output in {
            33,
            2,
            35,
            36,
            37,
            38,
            10,
            13,
            17,
            18,
            21,
            23,
            24,
            26,
            29,
        }:
            return attn_0_0_output == "1"
        elif mlp_0_0_output in {9, 30}:
            return attn_0_0_output == "<s>"
        elif mlp_0_0_output in {28, 14}:
            return attn_0_0_output == "<pad>"

    num_attn_2_7_pattern = select(attn_0_0_outputs, mlp_0_0_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_1_0_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_0_4_output):
        key = (attn_2_3_output, attn_0_4_output)
        return 27

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_0_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_0_output, mlp_0_3_output):
        key = (attn_2_0_output, mlp_0_3_output)
        return 39

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_0_outputs, mlp_0_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(num_mlp_0_1_output, attn_2_2_output):
        key = (num_mlp_0_1_output, attn_2_2_output)
        return 29

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(mlp_1_3_output, attn_0_4_output):
        key = (mlp_1_3_output, attn_0_4_output)
        return 32

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(mlp_1_3_outputs, attn_0_4_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_0_output, num_attn_1_5_output):
        key = (num_attn_2_0_output, num_attn_1_5_output)
        if key in {
            (47, 0),
            (48, 0),
            (49, 0),
            (50, 0),
            (51, 0),
            (52, 0),
            (53, 0),
            (54, 0),
            (55, 0),
            (56, 0),
            (57, 0),
            (58, 0),
            (59, 0),
            (60, 0),
            (61, 0),
            (61, 1),
            (62, 0),
            (62, 1),
            (63, 0),
            (63, 1),
            (64, 0),
            (64, 1),
            (65, 0),
            (65, 1),
            (66, 0),
            (66, 1),
            (67, 0),
            (67, 1),
            (68, 0),
            (68, 1),
            (69, 0),
            (69, 1),
            (70, 0),
            (70, 1),
            (71, 0),
            (71, 1),
            (72, 0),
            (72, 1),
            (73, 0),
            (73, 1),
            (74, 0),
            (74, 1),
            (75, 0),
            (75, 1),
            (76, 0),
            (76, 1),
            (76, 2),
            (77, 0),
            (77, 1),
            (77, 2),
            (78, 0),
            (78, 1),
            (78, 2),
            (79, 0),
            (79, 1),
            (79, 2),
            (80, 0),
            (80, 1),
            (80, 2),
            (81, 0),
            (81, 1),
            (81, 2),
            (82, 0),
            (82, 1),
            (82, 2),
            (83, 0),
            (83, 1),
            (83, 2),
            (84, 0),
            (84, 1),
            (84, 2),
            (85, 0),
            (85, 1),
            (85, 2),
            (86, 0),
            (86, 1),
            (86, 2),
            (87, 0),
            (87, 1),
            (87, 2),
            (88, 0),
            (88, 1),
            (88, 2),
            (89, 0),
            (89, 1),
            (89, 2),
            (90, 0),
            (90, 1),
            (90, 2),
            (90, 3),
            (91, 0),
            (91, 1),
            (91, 2),
            (91, 3),
            (92, 0),
            (92, 1),
            (92, 2),
            (92, 3),
            (93, 0),
            (93, 1),
            (93, 2),
            (93, 3),
            (94, 0),
            (94, 1),
            (94, 2),
            (94, 3),
            (95, 0),
            (95, 1),
            (95, 2),
            (95, 3),
            (96, 0),
            (96, 1),
            (96, 2),
            (96, 3),
            (97, 0),
            (97, 1),
            (97, 2),
            (97, 3),
            (98, 0),
            (98, 1),
            (98, 2),
            (98, 3),
            (99, 0),
            (99, 1),
            (99, 2),
            (99, 3),
            (100, 0),
            (100, 1),
            (100, 2),
            (100, 3),
            (101, 0),
            (101, 1),
            (101, 2),
            (101, 3),
            (102, 0),
            (102, 1),
            (102, 2),
            (102, 3),
            (103, 0),
            (103, 1),
            (103, 2),
            (103, 3),
            (104, 0),
            (104, 1),
            (104, 2),
            (104, 3),
            (104, 4),
            (105, 0),
            (105, 1),
            (105, 2),
            (105, 3),
            (105, 4),
            (106, 0),
            (106, 1),
            (106, 2),
            (106, 3),
            (106, 4),
            (107, 0),
            (107, 1),
            (107, 2),
            (107, 3),
            (107, 4),
            (108, 0),
            (108, 1),
            (108, 2),
            (108, 3),
            (108, 4),
            (109, 0),
            (109, 1),
            (109, 2),
            (109, 3),
            (109, 4),
            (110, 0),
            (110, 1),
            (110, 2),
            (110, 3),
            (110, 4),
            (111, 0),
            (111, 1),
            (111, 2),
            (111, 3),
            (111, 4),
            (112, 0),
            (112, 1),
            (112, 2),
            (112, 3),
            (112, 4),
            (113, 0),
            (113, 1),
            (113, 2),
            (113, 3),
            (113, 4),
            (114, 0),
            (114, 1),
            (114, 2),
            (114, 3),
            (114, 4),
            (115, 0),
            (115, 1),
            (115, 2),
            (115, 3),
            (115, 4),
            (116, 0),
            (116, 1),
            (116, 2),
            (116, 3),
            (116, 4),
            (117, 0),
            (117, 1),
            (117, 2),
            (117, 3),
            (117, 4),
            (118, 0),
            (118, 1),
            (118, 2),
            (118, 3),
            (118, 4),
            (118, 5),
            (119, 0),
            (119, 1),
            (119, 2),
            (119, 3),
            (119, 4),
            (119, 5),
        }:
            return 17
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_6_output, num_attn_0_4_output):
        key = (num_attn_2_6_output, num_attn_0_4_output)
        return 9

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_6_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_5_output, num_attn_1_7_output):
        key = (num_attn_1_5_output, num_attn_1_7_output)
        return 16

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output, num_attn_2_3_output):
        key = (num_attn_1_0_output, num_attn_2_3_output)
        return 20

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_3_outputs)
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


print(run(["<s>", "0", "1", "3", "0", "0", "0", "5", "5", "3", "2", "3"]))

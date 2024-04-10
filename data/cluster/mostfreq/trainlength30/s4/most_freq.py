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
        "output/length/rasp/mostfreq/trainlength30/s4/most_freq_weights.csv",
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
        if position in {0, 25, 26, 29}:
            return token == "4"
        elif position in {1, 2, 14, 16, 17, 19, 21, 22, 23, 24, 27, 28}:
            return token == "0"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {7, 39, 9, 10, 11, 12, 13, 15, 18, 20}:
            return token == "5"
        elif position in {8}:
            return token == "3"
        elif position in {32, 33, 34, 35, 36, 37, 38, 30, 31}:
            return token == ""

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 29}:
            return token == "3"
        elif position in {1, 12, 13, 14, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28}:
            return token == "0"
        elif position in {2, 3, 7}:
            return token == "2"
        elif position in {4, 5, 6}:
            return token == "4"
        elif position in {8, 9, 11, 15, 22, 24}:
            return token == "5"
        elif position in {32, 33, 34, 35, 36, 39, 10, 31}:
            return token == ""
        elif position in {23}:
            return token == "<s>"
        elif position in {38, 37, 30}:
            return token == "1"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 3
        elif q_position in {1, 10, 14, 15, 17, 18}:
            return k_position == 1
        elif q_position in {9, 2, 21}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 0
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {33, 6}:
            return k_position == 14
        elif q_position in {8, 12, 28, 7}:
            return k_position == 5
        elif q_position in {32, 11, 29}:
            return k_position == 22
        elif q_position in {13}:
            return k_position == 16
        elif q_position in {16}:
            return k_position == 11
        elif q_position in {24, 19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 24
        elif q_position in {22}:
            return k_position == 6
        elif q_position in {25, 30, 23}:
            return k_position == 28
        elif q_position in {26}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 17
        elif q_position in {31}:
            return k_position == 29
        elif q_position in {34, 38}:
            return k_position == 39
        elif q_position in {35, 36, 37}:
            return k_position == 26
        elif q_position in {39}:
            return k_position == 10

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 39}:
            return k_position == 7
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 14}:
            return k_position == 8
        elif q_position in {3, 31}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 18
        elif q_position in {5, 22}:
            return k_position == 17
        elif q_position in {32, 6}:
            return k_position == 24
        elif q_position in {8, 9, 19, 7}:
            return k_position == 4
        elif q_position in {10, 11, 13, 15, 27}:
            return k_position == 21
        elif q_position in {12, 18, 21, 24, 29}:
            return k_position == 25
        elif q_position in {16, 20, 23, 25, 26}:
            return k_position == 12
        elif q_position in {17}:
            return k_position == 3
        elif q_position in {28}:
            return k_position == 26
        elif q_position in {30}:
            return k_position == 16
        elif q_position in {33}:
            return k_position == 10
        elif q_position in {34}:
            return k_position == 34
        elif q_position in {35}:
            return k_position == 27
        elif q_position in {36}:
            return k_position == 28
        elif q_position in {37}:
            return k_position == 13
        elif q_position in {38}:
            return k_position == 31

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 8}:
            return k_position == 2
        elif q_position in {1, 27, 9}:
            return k_position == 1
        elif q_position in {2, 14, 7}:
            return k_position == 3
        elif q_position in {3, 20}:
            return k_position == 4
        elif q_position in {18, 11, 4, 28}:
            return k_position == 5
        elif q_position in {13, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {10, 29}:
            return k_position == 7
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {26, 15}:
            return k_position == 8
        elif q_position in {16}:
            return k_position == 29
        elif q_position in {17, 35}:
            return k_position == 24
        elif q_position in {32, 19, 39}:
            return k_position == 28
        elif q_position in {24, 21, 23}:
            return k_position == 0
        elif q_position in {22}:
            return k_position == 12
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {30}:
            return k_position == 32
        elif q_position in {31}:
            return k_position == 22
        elif q_position in {33}:
            return k_position == 14
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {36}:
            return k_position == 38
        elif q_position in {37}:
            return k_position == 16
        elif q_position in {38}:
            return k_position == 31

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 8, 9, 10, 12, 13, 16, 22, 25}:
            return token == "3"
        elif position in {1}:
            return token == "2"
        elif position in {2}:
            return token == "1"
        elif position in {3, 4, 5, 6, 36}:
            return token == "5"
        elif position in {7, 11, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29}:
            return token == "0"
        elif position in {32, 33, 34, 35, 37, 38, 39, 30, 31}:
            return token == ""

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1, 2, 3}:
            return token == "0"
        elif position in {4, 5, 6}:
            return token == "2"
        elif position in {7, 8, 10, 11, 12, 13, 17, 18, 20, 21, 23, 26, 27, 29}:
            return token == "5"
        elif position in {33, 34, 35, 36, 37, 38, 39, 9, 30, 31}:
            return token == ""
        elif position in {14, 22}:
            return token == "1"
        elif position in {24, 15}:
            return token == "<s>"
        elif position in {32, 16, 19, 25, 28}:
            return token == "4"

    attn_0_6_pattern = select_closest(tokens, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 33}:
            return k_position == 14
        elif q_position in {1, 7}:
            return k_position == 24
        elif q_position in {2, 31}:
            return k_position == 10
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 25
        elif q_position in {6}:
            return k_position == 18
        elif q_position in {8, 10, 11, 12, 13, 15, 17, 18, 21, 23, 25, 27, 28}:
            return k_position == 27
        elif q_position in {9, 19, 20, 14}:
            return k_position == 7
        elif q_position in {16, 26, 29, 22}:
            return k_position == 17
        elif q_position in {24}:
            return k_position == 12
        elif q_position in {30}:
            return k_position == 35
        elif q_position in {32}:
            return k_position == 22
        elif q_position in {34}:
            return k_position == 38
        elif q_position in {35}:
            return k_position == 33
        elif q_position in {36}:
            return k_position == 34
        elif q_position in {37}:
            return k_position == 30
        elif q_position in {38}:
            return k_position == 8
        elif q_position in {39}:
            return k_position == 37

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, positions)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 22}:
            return token == "<s>"
        elif position in {1}:
            return token == "0"
        elif position in {
            32,
            33,
            2,
            3,
            4,
            5,
            34,
            35,
            36,
            37,
            38,
            39,
            25,
            26,
            27,
            30,
            31,
        }:
            return token == ""
        elif position in {6, 16, 17, 21, 23, 24, 28, 29}:
            return token == "5"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 6}:
            return token == "3"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
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
        elif position in {29, 7}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 32, 2, 3, 4, 5, 33, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {1, 18, 11}:
            return token == "4"
        elif position in {6}:
            return token == "<pad>"
        elif position in {7, 9, 10, 15, 17, 19, 21, 22, 25}:
            return token == "5"
        elif position in {8, 13, 20, 24, 26, 28}:
            return token == "3"
        elif position in {27, 12, 23}:
            return token == "1"
        elif position in {16, 14}:
            return token == "0"
        elif position in {29}:
            return token == "2"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 7, 8, 9, 11}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {
            2,
            3,
            4,
            5,
            10,
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
        elif position in {6}:
            return token == "4"
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 1, 2, 3, 4}:
            return token == "2"
        elif position in {5, 6}:
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

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0, 4, 5, 6}:
            return token == "<s>"
        elif position in {1, 2, 3}:
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

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 6}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {
            2,
            3,
            4,
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
        if position in {0, 32, 2, 3, 33, 34, 6, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {
            1,
            8,
            9,
            10,
            11,
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
            28,
            29,
        }:
            return token == "5"
        elif position in {4, 5}:
            return token == "<pad>"
        elif position in {13, 7}:
            return token == "1"
        elif position in {12}:
            return token == "0"
        elif position in {27}:
            return token == "4"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        return 2

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_4_output):
        key = (attn_0_2_output, attn_0_4_output)
        return 1

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_6_output, attn_0_0_output):
        key = (attn_0_6_output, attn_0_0_output)
        return 3

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_0_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(position, attn_0_0_output):
        key = (position, attn_0_0_output)
        if key in {
            (0, "1"),
            (0, "2"),
            (0, "3"),
            (0, "4"),
            (0, "5"),
            (0, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
            (5, "1"),
            (5, "2"),
            (5, "3"),
            (5, "4"),
            (5, "5"),
            (5, "<s>"),
            (30, "1"),
            (30, "2"),
            (30, "3"),
            (30, "4"),
            (31, "1"),
            (31, "2"),
            (31, "3"),
            (31, "4"),
            (32, "1"),
            (32, "2"),
            (32, "3"),
            (32, "4"),
            (33, "1"),
            (33, "2"),
            (33, "3"),
            (33, "4"),
            (34, "1"),
            (34, "2"),
            (34, "3"),
            (34, "4"),
            (35, "1"),
            (35, "2"),
            (35, "3"),
            (35, "4"),
            (36, "1"),
            (36, "2"),
            (36, "3"),
            (36, "4"),
            (37, "1"),
            (37, "2"),
            (37, "3"),
            (37, "4"),
            (38, "1"),
            (38, "2"),
            (38, "3"),
            (38, "4"),
            (39, "1"),
            (39, "2"),
            (39, "3"),
            (39, "4"),
        }:
            return 20
        elif key in {
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
            (30, "<s>"),
            (33, "<s>"),
        }:
            return 4
        elif key in {(6, "1"), (6, "2"), (6, "3"), (6, "4")}:
            return 26
        return 34

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(positions, attn_0_0_outputs)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 12

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_6_output):
        key = (num_attn_0_1_output, num_attn_0_6_output)
        return 33

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 14

    num_mlp_0_2_outputs = [num_mlp_0_2(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_7_output):
        key = (num_attn_0_2_output, num_attn_0_7_output)
        return 11

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_7_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 32, 3, 4, 8, 9, 11, 13, 15, 31}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {2}:
            return token == "0"
        elif position in {5, 6, 12, 14, 16, 18, 20, 21, 22, 26}:
            return token == "3"
        elif position in {7}:
            return token == "2"
        elif position in {33, 34, 35, 36, 37, 38, 39, 10, 24, 25, 27, 28, 29, 30}:
            return token == ""
        elif position in {17, 23}:
            return token == "5"
        elif position in {19}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_5_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"3", "0", "2", "5", "<s>"}:
            return k_token == "4"
        elif q_token in {"4", "1"}:
            return k_token == "5"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 33, 32, 34, 35, 37, 30, 31}:
            return token == "4"
        elif position in {1}:
            return token == "2"
        elif position in {2, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 24, 26, 27, 28}:
            return token == "3"
        elif position in {3, 4, 5, 6, 11, 20, 22, 23, 25}:
            return token == "0"
        elif position in {12, 7}:
            return token == "5"
        elif position in {36, 29, 38, 39}:
            return token == ""

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 4, 5}:
            return token == "<s>"
        elif position in {1}:
            return token == "3"
        elif position in {2, 3}:
            return token == "5"
        elif position in {6, 7}:
            return token == "4"
        elif position in {
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
            20,
            21,
            22,
            23,
            25,
            26,
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
        elif position in {24, 17, 27}:
            return token == "0"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_token, k_token):
        if q_token in {"0", "1", "4"}:
            return k_token == "3"
        elif q_token in {"2", "5", "3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_1_4_pattern = select_closest(tokens, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(q_token, k_token):
        if q_token in {"4", "0", "1", "2", "5"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    attn_1_5_pattern = select_closest(tokens, tokens, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(token, position):
        if token in {"0", "5", "4"}:
            return position == 7
        elif token in {"1"}:
            return position == 1
        elif token in {"2"}:
            return position == 6
        elif token in {"3"}:
            return position == 22
        elif token in {"<s>"}:
            return position == 8

    attn_1_6_pattern = select_closest(positions, tokens, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_3_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(q_token, k_token):
        if q_token in {"4", "3", "0", "1", "2"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_1_7_pattern = select_closest(tokens, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, attn_0_5_output):
        if attn_0_0_output in {"4", "3", "0", "2", "5"}:
            return attn_0_5_output == ""
        elif attn_0_0_output in {"1"}:
            return attn_0_5_output == "1"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_5_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_5_outputs, attn_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, token):
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
            return token == "0"
        elif position in {5, 6}:
            return token == "1"

    num_attn_1_1_pattern = select(tokens, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, attn_0_0_output):
        if attn_0_1_output in {"0"}:
            return attn_0_0_output == "<pad>"
        elif attn_0_1_output in {"3", "2", "1", "5", "<s>"}:
            return attn_0_0_output == ""
        elif attn_0_1_output in {"4"}:
            return attn_0_0_output == "4"

    num_attn_1_2_pattern = select(attn_0_0_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_6_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(token, position):
        if token in {"0"}:
            return position == 26
        elif token in {"1"}:
            return position == 37
        elif token in {"2"}:
            return position == 7
        elif token in {"3"}:
            return position == 38
        elif token in {"4"}:
            return position == 14
        elif token in {"5"}:
            return position == 30
        elif token in {"<s>"}:
            return position == 0

    num_attn_1_3_pattern = select(positions, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, attn_0_6_output):
        if position in {0, 1}:
            return attn_0_6_output == "3"
        elif position in {
            2,
            3,
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
            18,
            19,
            20,
            21,
            22,
            23,
            25,
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
            return attn_0_6_output == ""
        elif position in {24, 17, 26, 27}:
            return attn_0_6_output == "<pad>"

    num_attn_1_4_pattern = select(attn_0_6_outputs, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(token, position):
        if token in {"0"}:
            return position == 28
        elif token in {"4", "1", "3"}:
            return position == 7
        elif token in {"2"}:
            return position == 32
        elif token in {"5"}:
            return position == 33
        elif token in {"<s>"}:
            return position == 0

    num_attn_1_5_pattern = select(positions, tokens, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_3_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, token):
        if position in {0, 6, 17, 18, 19, 25, 27}:
            return token == "<s>"
        elif position in {1}:
            return token == "1"
        elif position in {32, 33, 2, 3, 4, 5, 34, 35, 36, 37, 38, 39, 30, 31}:
            return token == ""
        elif position in {16, 21, 7}:
            return token == "3"
        elif position in {8, 9, 11, 12, 14, 15, 20, 26, 28, 29}:
            return token == "5"
        elif position in {24, 10, 22}:
            return token == "4"
        elif position in {13, 23}:
            return token == "0"

    num_attn_1_6_pattern = select(tokens, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, ones)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(token, attn_0_1_output):
        if token in {"0"}:
            return attn_0_1_output == "0"
        elif token in {"4", "3", "2", "1", "5", "<s>"}:
            return attn_0_1_output == ""

    num_attn_1_7_pattern = select(attn_0_1_outputs, tokens, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_6_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(mlp_0_3_output, num_mlp_0_2_output):
        key = (mlp_0_3_output, num_mlp_0_2_output)
        return 37

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(mlp_0_2_output, num_mlp_0_2_output):
        key = (mlp_0_2_output, num_mlp_0_2_output)
        return 1

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, num_mlp_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_0_output, attn_1_7_output):
        key = (attn_1_0_output, attn_1_7_output)
        return 6

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_7_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(num_mlp_0_3_output, mlp_0_1_output):
        key = (num_mlp_0_3_output, mlp_0_1_output)
        return 28

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(num_mlp_0_3_outputs, mlp_0_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_7_output, num_attn_1_5_output):
        key = (num_attn_1_7_output, num_attn_1_5_output)
        return 20

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_4_output, num_attn_1_0_output):
        key = (num_attn_1_4_output, num_attn_1_0_output)
        return 30

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_1_6_output, num_attn_1_4_output):
        key = (num_attn_1_6_output, num_attn_1_4_output)
        return 29

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_1_7_output, num_attn_1_4_output):
        key = (num_attn_1_7_output, num_attn_1_4_output)
        return 25

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0", "5", "3"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"0", "1"}:
            return token == "4"
        elif attn_0_1_output in {"2"}:
            return token == "3"
        elif attn_0_1_output in {"5", "3"}:
            return token == "1"
        elif attn_0_1_output in {"4", "<s>"}:
            return token == "5"

    attn_2_1_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_2_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"4", "3", "0", "2", "5"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_7_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0", "1"}:
            return k_token == "3"
        elif q_token in {"2", "5", "<s>"}:
            return k_token == ""
        elif q_token in {"4", "3"}:
            return k_token == "4"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_5_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_0_output, token):
        if attn_0_0_output in {"0", "2"}:
            return token == "1"
        elif attn_0_0_output in {"1", "5", "3"}:
            return token == "2"
        elif attn_0_0_output in {"4"}:
            return token == "3"
        elif attn_0_0_output in {"<s>"}:
            return token == ""

    attn_2_4_pattern = select_closest(tokens, attn_0_0_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_4_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_1_output, attn_1_1_output):
        if attn_0_1_output in {"0", "3", "4", "<s>"}:
            return attn_1_1_output == ""
        elif attn_0_1_output in {"1"}:
            return attn_1_1_output == "0"
        elif attn_0_1_output in {"2", "5"}:
            return attn_1_1_output == "4"

    attn_2_5_pattern = select_closest(attn_1_1_outputs, attn_0_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_1_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_0_0_output, mlp_0_3_output):
        if attn_0_0_output in {"0"}:
            return mlp_0_3_output == 21
        elif attn_0_0_output in {"1"}:
            return mlp_0_3_output == 4
        elif attn_0_0_output in {"2", "5"}:
            return mlp_0_3_output == 20
        elif attn_0_0_output in {"3"}:
            return mlp_0_3_output == 3
        elif attn_0_0_output in {"4"}:
            return mlp_0_3_output == 29
        elif attn_0_0_output in {"<s>"}:
            return mlp_0_3_output == 5

    attn_2_6_pattern = select_closest(mlp_0_3_outputs, attn_0_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_6_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, position):
        if token in {"0"}:
            return position == 3
        elif token in {"1"}:
            return position == 0
        elif token in {"2"}:
            return position == 20
        elif token in {"3"}:
            return position == 25
        elif token in {"4"}:
            return position == 6
        elif token in {"5"}:
            return position == 13
        elif token in {"<s>"}:
            return position == 7

    attn_2_7_pattern = select_closest(positions, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_6_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_6_output):
        if attn_1_2_output in {"0", "<s>"}:
            return attn_0_6_output == "0"
        elif attn_1_2_output in {"4", "3", "2", "1", "5"}:
            return attn_0_6_output == ""

    num_attn_2_0_pattern = select(attn_0_6_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_5_output, attn_0_4_output):
        if attn_0_5_output in {"4", "3", "0", "1", "2", "<s>"}:
            return attn_0_4_output == ""
        elif attn_0_5_output in {"5"}:
            return attn_0_4_output == "5"

    num_attn_2_1_pattern = select(attn_0_4_outputs, attn_0_5_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_5_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {"4", "3", "0", "2", "<s>"}:
            return k_attn_0_0_output == "1"
        elif q_attn_0_0_output in {"1"}:
            return k_attn_0_0_output == "4"
        elif q_attn_0_0_output in {"5"}:
            return k_attn_0_0_output == ""

    num_attn_2_2_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_5_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(position, attn_0_4_output):
        if position in {0, 1, 2, 3, 4, 33, 34, 35, 36, 37, 38, 39, 29, 30}:
            return attn_0_4_output == ""
        elif position in {5, 6, 14, 15, 17, 18, 21, 28}:
            return attn_0_4_output == "4"
        elif position in {7, 10, 12, 13, 16, 23, 25, 27}:
            return attn_0_4_output == "5"
        elif position in {8}:
            return attn_0_4_output == "3"
        elif position in {9, 11, 19, 20, 22, 24}:
            return attn_0_4_output == "0"
        elif position in {26}:
            return attn_0_4_output == "1"
        elif position in {32, 31}:
            return attn_0_4_output == "<pad>"

    num_attn_2_3_pattern = select(attn_0_4_outputs, positions, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_1_output, attn_0_4_output):
        if attn_0_1_output in {"0", "1", "5"}:
            return attn_0_4_output == "4"
        elif attn_0_1_output in {"2", "3", "<s>"}:
            return attn_0_4_output == ""
        elif attn_0_1_output in {"4"}:
            return attn_0_4_output == "3"

    num_attn_2_4_pattern = select(attn_0_4_outputs, attn_0_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_4_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(position, token):
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
            return token == "4"
        elif position in {5, 6}:
            return token == "0"

    num_attn_2_5_pattern = select(tokens, positions, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, ones)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_2_output, attn_1_3_output):
        if attn_0_2_output in {"4", "0", "1", "2", "5"}:
            return attn_1_3_output == ""
        elif attn_0_2_output in {"3", "<s>"}:
            return attn_1_3_output == "3"

    num_attn_2_6_pattern = select(attn_1_3_outputs, attn_0_2_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_4_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_6_output, attn_0_4_output):
        if attn_0_6_output in {"4", "3", "0", "1", "5", "<s>"}:
            return attn_0_4_output == ""
        elif attn_0_6_output in {"2"}:
            return attn_0_4_output == "2"

    num_attn_2_7_pattern = select(attn_0_4_outputs, attn_0_6_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_4_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_7_output, attn_0_4_output):
        key = (attn_0_7_output, attn_0_4_output)
        return 3

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_0_4_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_1_2_output, mlp_1_2_output):
        key = (num_mlp_1_2_output, mlp_1_2_output)
        if key in {
            (0, 3),
            (1, 3),
            (1, 14),
            (1, 29),
            (7, 1),
            (7, 3),
            (7, 5),
            (7, 14),
            (7, 29),
            (7, 35),
            (30, 1),
            (30, 3),
            (30, 5),
            (30, 14),
            (30, 29),
            (30, 35),
            (33, 1),
            (33, 3),
            (33, 5),
            (33, 14),
            (33, 29),
            (33, 33),
            (33, 35),
            (34, 1),
            (34, 3),
            (34, 5),
            (34, 14),
            (34, 16),
            (34, 21),
            (34, 29),
            (34, 32),
            (34, 33),
            (34, 35),
            (35, 1),
            (35, 3),
            (35, 5),
            (35, 14),
            (35, 16),
            (35, 29),
            (35, 33),
            (35, 35),
            (39, 3),
        }:
            return 32
        elif key in {
            (13, 8),
            (13, 18),
            (13, 36),
            (13, 39),
            (33, 8),
            (33, 16),
            (33, 21),
            (33, 32),
            (33, 39),
            (37, 8),
            (37, 36),
            (37, 39),
        }:
            return 37
        return 5

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_1_2_outputs, mlp_1_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(num_mlp_0_1_output, attn_0_1_output):
        key = (num_mlp_0_1_output, attn_0_1_output)
        return 21

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_0_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(num_mlp_0_2_output, attn_1_2_output):
        key = (num_mlp_0_2_output, attn_1_2_output)
        return 36

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(num_mlp_0_2_outputs, attn_1_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_1_output):
        key = num_attn_1_1_output
        return 25

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_1_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_6_output):
        key = num_attn_2_6_output
        return 24

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_6_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_2_0_output, num_attn_2_4_output):
        key = (num_attn_2_0_output, num_attn_2_4_output)
        return 21

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_4_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output, num_attn_2_6_output):
        key = (num_attn_1_0_output, num_attn_2_6_output)
        return 8

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_6_outputs)
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


print(
    run(
        [
            "<s>",
            "5",
            "1",
            "0",
            "0",
            "2",
            "1",
            "2",
            "4",
            "5",
            "1",
            "0",
            "4",
            "2",
            "4",
            "2",
            "4",
            "3",
            "0",
            "5",
            "5",
            "1",
            "5",
            "0",
            "2",
            "5",
            "0",
            "1",
        ]
    )
)
